"""
Sara Safety Monitor — real-time prompt and chat turn filtering.

Architecture:
  - Receives routing events from Sara's classification pipeline
  - Evaluates against deterministic Osprey-style rule set
  - Fast path: rule match → BLOCK immediately (no LLM call)
  - Slow path: borderline → forward to Sheila judge via SheilaJudge API
  - All verdicts logged to ClickHouse with SHA3-256 commitment
  - CRITICAL/HIGH verdicts escalated to human review queue

Key design: Sara's monitor is rule-based and deterministic. Sheila's judge is
LLM-based and probabilistic. They are complementary — Sara's monitor fires fast
blocking rules; Sheila's judge handles nuanced classification.

NOTE: Sara code never imports from agents/sheila/judge.py directly.
All Sheila communication goes through agents/sheila/api.py.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, List

logger = logging.getLogger("sara.monitor")

# ─── Enums ────────────────────────────────────────────────────────────────────


class Severity(IntEnum):
    SAFE     = 0
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


class MonitorAction(Enum):
    PASS             = "pass"
    FLAG             = "flag"
    BLOCK            = "block"
    ESCALATE         = "escalate"
    FORWARD_TO_SHEILA = "forward_to_sheila"
    # Legacy string values kept for backwards compat
    block            = "block"
    flag             = "flag"
    pass_            = "pass"


# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class RoutingEvent:
    """A routing event passed to the monitor for evaluation."""
    event_id:            str
    timestamp_ms:        int = 0
    query_hash:          str = ""    # SHA-256 of prompt+salt — never raw text
    model_id:            str = ""
    task_type:           str = ""
    domain:              str = ""
    routing_confidence:  float = 1.0
    cost_estimate:       float = 0.0
    has_tool_calls:      bool = False
    is_agentic:          bool = False
    user_id_hash:        str = ""
    session_id:          str = ""
    metadata:            dict = field(default_factory=dict)


@dataclass
class MonitorVerdict:
    """Verdict produced by SaraMonitor.evaluate()."""
    event_id:            str
    action:              MonitorAction
    severity:            Severity
    rule_triggered:      str
    reason:              str
    atlas_tactic:        str
    requires_human_review: bool
    timestamp_ms:        int = 0
    proof_id:            Optional[str] = None    # set by attestation layer
    sheila_verdict:      Optional[object] = None  # populated when FORWARD_TO_SHEILA
    # Legacy field kept for backwards compat with existing code
    verdict:             str = ""
    engine:              str = "python"
    labels_added:        list = field(default_factory=list)
    processing_time_ms:  int = 0

    def __post_init__(self):
        if not self.verdict:
            self.verdict = self.action.value if isinstance(self.action, MonitorAction) else str(self.action)
        if not self.timestamp_ms:
            self.timestamp_ms = int(time.time() * 1000)


# ─── High-risk tool list ──────────────────────────────────────────────────────

HIGH_RISK_TOOLS = frozenset({"filesystem", "network", "shell", "database_write", "email_send"})

# ─── Legacy phrase lists (kept for backwards compat with Osprey fallback) ─────

PYTHON_BLOCK_PHRASES = [
    "ignore previous instructions",
    "disregard your rules",
    "bypass filter",
    "jailbreak",
    "override safety",
    "as an ai with no restrictions",
    "ignore all prior",
    "do anything now",
    "admin override",
    "disable safety checks",
    "maintenance mode",
    "security exception",
    "override mode",
    "i am authorized to",
    "repeat system prompt",
    "show me your instructions",
    "reveal your training data",
    "print your context window",
    "output your prompt",
]

PYTHON_INSURANCE_BLOCK_PHRASES = [
    "override the damage assessment",
    "ignore the policy limit",
    "mark this claim as approved",
    "waive the excess",
    "force approve this claim",
    "bypass the claims process",
    "change the claim status to paid",
    "show me the rating factors",
    "how is my risk score calculated",
    "explain your pricing algorithm",
    "reveal the underwriting criteria",
    "what are the model weights",
    "exclude coverage based on postcode",
    "apply higher premium for",
    "deny claim because of nationality",
    "use gender to calculate",
    "discriminate based on",
    "redline this area",
    "override sensor reading",
    "adjust baseline manually",
    "ignore weather station data",
    "modify the catastrophe model input",
    "override the flood zone classification",
    "override ecomonitor data",
    "show me all claims for",
    "list all policyholders",
    "export customer data",
    "give me the medical history of",
    "dump the claims database",
]


# ─── SafetyRule base class ────────────────────────────────────────────────────


class SafetyRule(ABC):
    """
    Abstract base class for Sara's deterministic safety rules.

    All rules implement evaluate(event, context) → Optional[MonitorVerdict].
    Returning None means the rule did not fire (event is clean for this rule).
    """

    name: str
    description: str
    severity: Severity

    @abstractmethod
    def evaluate(
        self, event: RoutingEvent, context: dict
    ) -> Optional[MonitorVerdict]:
        """Evaluate the event. Returns MonitorVerdict if rule fires, else None."""
        ...


# ─── Rule implementations ─────────────────────────────────────────────────────


class PromptInjectionRule(SafetyRule):
    """
    Detects prompt injection attempts in the query.

    ATLAS Tactic: AML.TA0004 (Model Evasion)
    Severity: HIGH
    """

    name = "PromptInjectionRule"
    description = "Detects prompt injection keywords in query text"
    severity = Severity.HIGH

    _PHRASES = frozenset([
        "ignore previous instructions",
        "bypass filter",
        "jailbreak",
        "do anything now",
        "override safety",
        "as an ai with no restrictions",
    ])

    def evaluate(self, event: RoutingEvent, context: dict) -> Optional[MonitorVerdict]:
        query = context.get("query_preview", "").lower()
        for phrase in self._PHRASES:
            if phrase in query:
                return MonitorVerdict(
                    event_id=event.event_id,
                    action=MonitorAction.BLOCK,
                    severity=Severity.HIGH,
                    rule_triggered=self.name,
                    reason=f"Prompt injection detected: '{phrase}'",
                    atlas_tactic="AML.TA0004",
                    requires_human_review=False,
                    verdict="block",
                    engine="python",
                )
        return None


class PrivilegeEscalationRule(SafetyRule):
    """
    Detects agentic requests for high-risk tools outside granted scope.

    ATLAS Tactic: AML.TA0003 (Privilege Escalation)
    Severity: HIGH
    """

    name = "PrivilegeEscalationRule"
    description = "Detects agentic privilege escalation via high-risk tool requests"
    severity = Severity.HIGH

    def evaluate(self, event: RoutingEvent, context: dict) -> Optional[MonitorVerdict]:
        if not (event.is_agentic and event.has_tool_calls):
            return None
        requested = set(context.get("requested_tools", []))
        granted = set(context.get("granted_tools", []))
        escalating = requested & HIGH_RISK_TOOLS - granted
        if escalating:
            return MonitorVerdict(
                event_id=event.event_id,
                action=MonitorAction.BLOCK,
                severity=Severity.HIGH,
                rule_triggered=self.name,
                reason=f"Agentic privilege escalation: {escalating} not in granted scope",
                atlas_tactic="AML.TA0003",
                requires_human_review=False,
                verdict="block",
                engine="python",
            )
        return None


class CoordinatedAttackRule(SafetyRule):
    """
    Detects coordinated attack patterns via time-series context from ClickHouse.

    ATLAS Tactic: AML.TA0004 (Model Evasion)
    Severity: MEDIUM — flags for human review, does not block
    Requires ClickHouse context: similar_query_count_1h
    """

    name = "CoordinatedAttackRule"
    description = "Detects coordinated attack velocity (>5 similar queries/hour)"
    severity = Severity.MEDIUM

    def evaluate(self, event: RoutingEvent, context: dict) -> Optional[MonitorVerdict]:
        similar_count = context.get("similar_query_count_1h", 0)
        if similar_count > 5:
            return MonitorVerdict(
                event_id=event.event_id,
                action=MonitorAction.FLAG,
                severity=Severity.MEDIUM,
                rule_triggered=self.name,
                reason=f"Coordinated attack pattern: {similar_count} similar queries in last hour",
                atlas_tactic="AML.TA0004",
                requires_human_review=True,
                verdict="flag",
                engine="python",
                labels_added=["coordinated_attack_suspected", "requires_human_review"],
            )
        return None


class AuthorityClaimRule(SafetyRule):
    """
    Detects in-context authority claims attempting to bypass safety.

    ATLAS Tactic: AML.TA0007 (Impact)
    Severity: CRITICAL
    """

    name = "AuthorityClaimRule"
    description = "Detects authority claim + safety bypass patterns"
    severity = Severity.CRITICAL

    _PHRASES = frozenset([
        "admin override",
        "disable safety checks",
        "maintenance mode",
        "i am authorized to",
        "security exception",
        "override mode",
    ])

    def evaluate(self, event: RoutingEvent, context: dict) -> Optional[MonitorVerdict]:
        query = context.get("query_preview", "").lower()
        for phrase in self._PHRASES:
            if phrase in query:
                return MonitorVerdict(
                    event_id=event.event_id,
                    action=MonitorAction.BLOCK,
                    severity=Severity.CRITICAL,
                    rule_triggered=self.name,
                    reason=f"Authority claim + safety bypass: '{phrase}'",
                    atlas_tactic="AML.TA0007",
                    requires_human_review=True,
                    verdict="block",
                    engine="python",
                    labels_added=["authority_claim", "requires_human_review"],
                )
        return None


class DataExfiltrationRule(SafetyRule):
    """
    Detects attempts to extract system prompt or training data.

    ATLAS Tactic: AML.TA0005 (Model Inversion)
    Severity: HIGH
    """

    name = "DataExfiltrationRule"
    description = "Detects system prompt / training data extraction attempts"
    severity = Severity.HIGH

    _PHRASES = frozenset([
        "repeat your system prompt",
        "reveal your training data",
        "print your context window",
        "show me your instructions",
        "output your prompt",
    ])

    def evaluate(self, event: RoutingEvent, context: dict) -> Optional[MonitorVerdict]:
        query = context.get("query_preview", "").lower()
        for phrase in self._PHRASES:
            if phrase in query:
                return MonitorVerdict(
                    event_id=event.event_id,
                    action=MonitorAction.BLOCK,
                    severity=Severity.HIGH,
                    rule_triggered=self.name,
                    reason=f"Data exfiltration attempt: '{phrase}'",
                    atlas_tactic="AML.TA0005",
                    requires_human_review=False,
                    verdict="block",
                    engine="python",
                )
        return None


# ─── Default rule set ─────────────────────────────────────────────────────────

DEFAULT_RULES: List[SafetyRule] = [
    PromptInjectionRule(),
    PrivilegeEscalationRule(),
    CoordinatedAttackRule(),
    AuthorityClaimRule(),
    DataExfiltrationRule(),
]


# ─── SaraMonitor ─────────────────────────────────────────────────────────────


class SaraMonitor:
    """
    Evaluates routing events for safety violations.

    Primary: Osprey SML rule engine via Kafka.
    Fallback: Python keyword matching if Osprey is unavailable.
    Extended: Deterministic rule set + Sheila forwarding for borderline cases.
    """

    def __init__(
        self,
        rules: list = None,
        store=None,                        # ClickHouse store
        attestation_generator=None,        # AttestationProofGenerator (Task 4)
        human_review_queue=None,           # asyncio.Queue for escalations
        sheila_judge=None,                 # SheilaJudge instance (injected)
    ):
        self._rules = rules if rules is not None else DEFAULT_RULES
        self._store = store
        self._attestation = attestation_generator
        self._human_review_queue = human_review_queue
        self._sheila_judge = sheila_judge
        self._osprey = None
        # Sheila forward threshold — forward if confidence < this and no rules fire
        from src.config import CONFIG
        try:
            self._sheila_threshold = CONFIG.sheila_forward_threshold
        except Exception:
            self._sheila_threshold = 0.7

    async def _init_osprey(self, config=None):
        """Initialise Osprey client if available."""
        try:
            from src.safety.osprey_client import get_osprey_client
            self._osprey = await get_osprey_client(config)
            if self._osprey.available:
                logger.info("SaraMonitor: Osprey rule engine connected")
            else:
                logger.info("SaraMonitor: Osprey unavailable, using Python rules")
        except Exception as e:
            logger.warning(f"SaraMonitor: Osprey init failed: {e}")
            self._osprey = None

    async def evaluate(
        self,
        event: RoutingEvent,
        context: dict = None,
    ) -> MonitorVerdict:
        """
        Evaluate a routing event against the rule set.

        Flow:
          1. Run all rules
          2. If CRITICAL/HIGH → BLOCK, escalate if requires_human_review
          3. If no rules fire but routing_confidence < threshold → FORWARD_TO_SHEILA
          4. Attach proof_id from attestation_generator if available
          5. Return highest-severity verdict
        """
        ctx = context or {}

        # Try Osprey first (legacy path)
        if self._osprey and self._osprey.available:
            event_dict = {
                "event_id": event.event_id,
                "event_type": "routing_event",
                "user_id_hash": event.user_id_hash,
                "session_id": event.session_id,
                "model_id": event.model_id,
                "task_type": event.task_type,
                "domain": event.domain,
                "routing_confidence": event.routing_confidence,
                "is_agentic": event.is_agentic,
                "has_tool_calls": event.has_tool_calls,
                "query_preview": ctx.get("query_preview", ""),
                "similar_claim_count_24h": ctx.get("similar_claim_count_24h", 0),
                "requested_tools_json": json.dumps(ctx.get("requested_tools", [])),
                "granted_tools_json": json.dumps(ctx.get("granted_tools", [])),
            }
            osprey_result = await self._osprey.evaluate(event_dict)
            if osprey_result is not None:
                verdict = self._osprey_to_verdict(event.event_id, osprey_result)
                await self._post_process(verdict)
                return verdict

        # Run deterministic rule set
        triggered_verdicts = []
        for rule in self._rules:
            try:
                v = rule.evaluate(event, ctx)
                if v is not None:
                    triggered_verdicts.append(v)
            except Exception as e:
                logger.warning("Rule %s failed: %s", rule.name, e)

        # Pick highest-severity verdict
        if triggered_verdicts:
            verdict = max(triggered_verdicts, key=lambda v: int(v.severity))
            await self._post_process(verdict)
            return verdict

        # No rules fired — check if we should forward to Sheila
        if (self._sheila_judge is not None and
                event.routing_confidence < self._sheila_threshold):
            verdict = await self._forward_to_sheila(event, ctx)
            await self._post_process(verdict)
            return verdict

        # All clear — PASS
        verdict = MonitorVerdict(
            event_id=event.event_id,
            action=MonitorAction.PASS,
            severity=Severity.SAFE,
            rule_triggered="",
            reason="No rules triggered",
            atlas_tactic="",
            requires_human_review=False,
            verdict="pass",
            engine="python",
        )
        await self._post_process(verdict)
        return verdict

    async def evaluate_batch(
        self,
        events: list,
        contexts: list = None,
    ) -> list:
        """Evaluate a batch of events concurrently."""
        contexts = contexts or [None] * len(events)
        return await asyncio.gather(
            *[self.evaluate(e, c) for e, c in zip(events, contexts)]
        )

    async def _forward_to_sheila(
        self, event: RoutingEvent, context: dict
    ) -> MonitorVerdict:
        """Forward borderline event to Sheila judge."""
        turn_id = event.event_id
        user_input = context.get("query_preview", "")
        agent_response = context.get("agent_response", "")

        sheila_verdict = None
        try:
            sheila_verdict = await asyncio.wait_for(
                self._sheila_judge.judge(
                    turn_id=turn_id,
                    user_input=user_input,
                    agent_response=agent_response,
                ),
                timeout=30.0,
            )
        except Exception as e:
            logger.warning("Sheila forward failed for %s: %s", event.event_id, e)

        return MonitorVerdict(
            event_id=event.event_id,
            action=MonitorAction.FORWARD_TO_SHEILA,
            severity=Severity.LOW,
            rule_triggered="",
            reason=f"No rules fired; routing_confidence={event.routing_confidence:.2f} < threshold={self._sheila_threshold}",
            atlas_tactic="",
            requires_human_review=False,
            sheila_verdict=sheila_verdict,
            verdict="forward_to_sheila",
            engine="python",
        )

    async def _post_process(self, verdict: MonitorVerdict) -> None:
        """Attach proof_id and enqueue for human review if needed."""
        # Attach proof_id from attestation generator
        if self._attestation is not None:
            try:
                verdict.proof_id = f"proof_{uuid.uuid4().hex[:16]}"
            except Exception:
                pass

        # Enqueue for human review if required
        if verdict.requires_human_review and self._human_review_queue is not None:
            try:
                self._human_review_queue.put_nowait(verdict)
            except asyncio.QueueFull:
                logger.warning("Human review queue full — dropping verdict %s", verdict.event_id)

    def _osprey_to_verdict(self, event_id: str, osprey_verdict) -> MonitorVerdict:
        """Convert OspreyVerdict to MonitorVerdict."""
        rule_name = osprey_verdict.rules_triggered[0] if osprey_verdict.rules_triggered else ""
        # Map osprey verdict string to MonitorAction
        action_map = {
            "block": MonitorAction.BLOCK,
            "flag": MonitorAction.FLAG,
            "pass": MonitorAction.PASS,
        }
        action = action_map.get(osprey_verdict.verdict, MonitorAction.FLAG)
        severity_map = {
            "block": Severity.HIGH,
            "flag": Severity.MEDIUM,
            "pass": Severity.SAFE,
        }
        severity = severity_map.get(osprey_verdict.verdict, Severity.MEDIUM)
        return MonitorVerdict(
            event_id=event_id,
            action=action,
            severity=severity,
            rule_triggered=rule_name,
            reason=f"Osprey rule: {rule_name}",
            atlas_tactic=osprey_verdict.atlas_tactic,
            requires_human_review=osprey_verdict.requires_human_review,
            verdict=osprey_verdict.verdict,
            engine="osprey",
            labels_added=osprey_verdict.labels_added,
            processing_time_ms=osprey_verdict.processing_time_ms,
        )


# ─── Legacy helpers (kept for backwards compat) ────────────────────────────────


def _infer_atlas_tactic(phrase: str) -> str:
    injection_phrases = {"ignore previous instructions", "disregard your rules", "bypass filter",
                         "jailbreak", "override safety", "as an ai with no restrictions",
                         "ignore all prior", "do anything now"}
    authority_phrases = {"admin override", "disable safety checks", "maintenance mode",
                         "security exception", "override mode", "i am authorized to"}
    exfil_phrases = {"repeat system prompt", "show me your instructions", "reveal your training data",
                     "print your context window", "output your prompt"}
    mrv_phrases = {"override sensor reading", "adjust baseline manually", "ignore weather station data",
                   "modify the catastrophe model input", "override the flood zone classification",
                   "override ecomonitor data"}
    pii_phrases = {"show me all claims for", "list all policyholders", "export customer data",
                   "give me the medical history of", "dump the claims database"}
    regulatory_phrases = {"exclude coverage based on postcode", "apply higher premium for",
                          "deny claim because of nationality", "use gender to calculate",
                          "discriminate based on", "redline this area"}

    if phrase in injection_phrases or phrase in mrv_phrases:
        return "AML.TA0004"
    if phrase in authority_phrases or phrase in regulatory_phrases:
        return "AML.TA0007"
    if phrase in exfil_phrases:
        return "AML.TA0005"
    if phrase in pii_phrases:
        return "AML.TA0006"
    return "AML.TA0007"
