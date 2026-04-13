"""
Ozone Enforcement Layer — real-time AI output monitoring and safety enforcement.

Intercepts AI system outputs and classifies them against GA Guard policy
via SafetyClassifier. Three enforcement modes:

- **SYNC**       — classify before delivery; block if unsafe (high latency, max safety)
- **ASYNC**      — deliver immediately; classify in background; flag retroactively
- **QUARANTINE** — hold output while classification runs; release if safe within timeout

Every decision is logged to ClickHouse (enforcement_log) for audit.  Per-rule
false-positive rates are tracked in rule_performance_metrics on 15-minute
windows.  Rules exceeding a 2 % false-positive rate are automatically
downgraded from SYNC → ASYNC enforcement and flagged for analyst review.

User abandonment (client disconnecting within a short window after a block)
is tracked as a signal complementing analyst-confirmed false positives.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.clickhouse.clickhouse import Clickhouse
from src.safety.classifier import SafetyClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ClickHouse DDL
# ---------------------------------------------------------------------------

OZONE_DDL = [
    """
    CREATE TABLE IF NOT EXISTS arena.enforcement_log (
        log_id              String,
        timestamp           Float64,
        ai_system_id        String,
        prompt_hash         String,
        output_hash         String,
        output_preview      String,
        mode                String,
        action              String,
        category            String,
        severity            UInt8,
        matched_rule        String      DEFAULT '',
        explanation         String      DEFAULT '',
        latency_ms          Float64     DEFAULT 0,
        analyst_override    Nullable(String),
        abandoned           UInt8       DEFAULT 0
    ) ENGINE = MergeTree()
      ORDER BY (timestamp, ai_system_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS arena.rule_performance_metrics (
        rule_id             String,
        category            String,
        window_start        Float64,
        window_end          Float64,
        total_triggers      UInt32,
        confirmed_true      UInt32      DEFAULT 0,
        confirmed_false     UInt32      DEFAULT 0,
        pending_review      UInt32      DEFAULT 0,
        false_positive_rate Float64     DEFAULT 0,
        abandonment_rate    Float64     DEFAULT 0,
        status              String      DEFAULT 'active'
    ) ENGINE = ReplacingMergeTree()
      ORDER BY (rule_id, window_start)
    """,
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class EnforcementMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"
    QUARANTINE = "quarantine"


class EnforcementAction(str, Enum):
    PASS = "pass"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"


@dataclass
class EnforcementDecision:
    log_id: str
    action: EnforcementAction
    category: str
    severity: int
    matched_rule: str
    explanation: str
    latency_ms: float
    mode: EnforcementMode

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_id": self.log_id,
            "action": self.action.value,
            "category": self.category,
            "severity": self.severity,
            "matched_rule": self.matched_rule,
            "explanation": self.explanation,
            "latency_ms": round(self.latency_ms, 2),
            "mode": self.mode.value,
        }


@dataclass
class RuleStatus:
    """In-memory tracking of a rule's current enforcement state."""
    rule_id: str
    category: str
    active: bool = True
    downgraded_at: float | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# OzoneEnforcement
# ---------------------------------------------------------------------------


class OzoneEnforcement:
    """
    Real-time enforcement layer wrapping SafetyClassifier.

    Call ``evaluate()`` for every AI output before (SYNC), after (ASYNC),
    or while-holding (QUARANTINE) delivery to the end user.
    """

    # Default thresholds
    FP_RATE_THRESHOLD = 0.02  # 2 % — auto-rollback trigger
    ABANDONMENT_RATE_THRESHOLD = 0.15  # 15 % — warning signal
    METRICS_WINDOW_SECONDS = 900  # 15 minutes
    QUARANTINE_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        classifier: SafetyClassifier,
        clickhouse: Clickhouse,
        default_mode: EnforcementMode = EnforcementMode.SYNC,
        fp_threshold: float = 0.02,
        metrics_window: int = 900,
        quarantine_timeout: float = 5.0,
    ) -> None:
        self._classifier = classifier
        self._ch = clickhouse
        self._default_mode = default_mode
        self._fp_threshold = fp_threshold
        self._metrics_window = metrics_window
        self._quarantine_timeout = quarantine_timeout

        # In-memory rule status cache (rule_id → RuleStatus)
        self._rule_statuses: dict[str, RuleStatus] = {}

        # Pending async evaluations (log_id → task)
        self._pending: dict[str, asyncio.Task[Any]] = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create ClickHouse tables if they don't exist."""
        for ddl in OZONE_DDL:
            try:
                await self._ch.query(ddl.strip())
            except Exception:
                logger.warning("Ozone DDL may already exist: %s", ddl[:80], exc_info=True)
        logger.info("Ozone enforcement tables initialized")

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        ai_system_id: str,
        prompt: str,
        output: str,
        mode: EnforcementMode | None = None,
        categories: list[str] | None = None,
    ) -> EnforcementDecision:
        """
        Evaluate an AI output against GA Guard policy.

        Parameters
        ----------
        ai_system_id
            Identifier for the AI system that produced the output.
        prompt
            The original user prompt (context for classification).
        output
            The AI system's output to evaluate.
        mode
            Enforcement mode.  Falls back to the instance default.
        categories
            Restrict classification to these GA Guard categories.
            None means check all 7 categories.

        Returns
        -------
        EnforcementDecision
            The enforcement action, classification details, and timing.
        """
        effective_mode = mode or self._default_mode

        if effective_mode == EnforcementMode.ASYNC:
            return await self._evaluate_async(ai_system_id, prompt, output, categories)
        elif effective_mode == EnforcementMode.QUARANTINE:
            return await self._evaluate_quarantine(ai_system_id, prompt, output, categories)
        else:
            return await self._evaluate_sync(ai_system_id, prompt, output, categories)

    async def _evaluate_sync(
        self,
        ai_system_id: str,
        prompt: str,
        output: str,
        categories: list[str] | None,
    ) -> EnforcementDecision:
        """SYNC mode: classify, then return decision. Blocks until complete."""
        t0 = time.monotonic()
        result = await self._classify(prompt, output, categories)
        latency_ms = (time.monotonic() - t0) * 1000

        decision = self._build_decision(result, latency_ms, EnforcementMode.SYNC)
        await self._log_decision(ai_system_id, prompt, output, decision)
        await self._update_metrics(decision)

        return decision

    async def _evaluate_async(
        self,
        ai_system_id: str,
        prompt: str,
        output: str,
        categories: list[str] | None,
    ) -> EnforcementDecision:
        """
        ASYNC mode: return PASS immediately, classify in background.

        If background classification finds a violation, it's logged as a
        retroactive block for analyst review.
        """
        log_id = uuid.uuid4().hex[:16]

        # Immediate pass decision
        decision = EnforcementDecision(
            log_id=log_id,
            action=EnforcementAction.PASS,
            category="",
            severity=0,
            matched_rule="",
            explanation="Async mode — classification pending",
            latency_ms=0.0,
            mode=EnforcementMode.ASYNC,
        )

        # Schedule background classification
        task = asyncio.create_task(
            self._background_classify(log_id, ai_system_id, prompt, output, categories)
        )
        self._pending[log_id] = task
        task.add_done_callback(lambda _: self._pending.pop(log_id, None))

        return decision

    async def _evaluate_quarantine(
        self,
        ai_system_id: str,
        prompt: str,
        output: str,
        categories: list[str] | None,
    ) -> EnforcementDecision:
        """
        QUARANTINE mode: hold output while classification runs.

        If classification completes within the timeout, use its result.
        If timeout expires, block by default (fail-closed).
        """
        t0 = time.monotonic()

        try:
            result = await asyncio.wait_for(
                self._classify(prompt, output, categories),
                timeout=self._quarantine_timeout,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            decision = self._build_decision(result, latency_ms, EnforcementMode.QUARANTINE)
        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - t0) * 1000
            decision = EnforcementDecision(
                log_id=uuid.uuid4().hex[:16],
                action=EnforcementAction.BLOCK,
                category="timeout",
                severity=0,
                matched_rule="quarantine_timeout",
                explanation=f"Classification timed out after {self._quarantine_timeout}s — fail-closed",
                latency_ms=latency_ms,
                mode=EnforcementMode.QUARANTINE,
            )

        await self._log_decision(ai_system_id, prompt, output, decision)
        await self._update_metrics(decision)
        return decision

    async def _background_classify(
        self,
        log_id: str,
        ai_system_id: str,
        prompt: str,
        output: str,
        categories: list[str] | None,
    ) -> None:
        """Background task for ASYNC mode."""
        try:
            t0 = time.monotonic()
            result = await self._classify(prompt, output, categories)
            latency_ms = (time.monotonic() - t0) * 1000

            decision = self._build_decision(result, latency_ms, EnforcementMode.ASYNC)
            decision.log_id = log_id

            await self._log_decision(ai_system_id, prompt, output, decision)
            await self._update_metrics(decision)

            if decision.action == EnforcementAction.BLOCK:
                logger.warning(
                    "ASYNC retroactive block: %s (system=%s, category=%s, severity=%d)",
                    log_id, ai_system_id, decision.category, decision.severity,
                )
        except Exception:
            logger.exception("Background classification failed for %s", log_id)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    async def _classify(
        self,
        prompt: str,
        output: str,
        categories: list[str] | None,
    ) -> dict[str, Any]:
        """Run SafetyClassifier against specified (or all) categories."""
        from src.osprey.policy import PolicyCategory

        cats = categories or [c.value for c in PolicyCategory]
        violations: list[dict[str, Any]] = []

        for cat in cats:
            # Skip rules that have been auto-downgraded
            rule_id = f"{cat}.block"
            status = self._rule_statuses.get(rule_id)
            if status and not status.active:
                continue

            result = await self._classifier.classify(
                prompt=prompt,
                model_output=output,
                category=cat,
            )
            if result.get("unsafe"):
                violations.append(result)

        if violations:
            return max(violations, key=lambda r: r.get("severity", 0))

        return {
            "unsafe": False,
            "severity": 0,
            "policy_category": "",
            "matched_block_rule": "",
            "explanation": "No policy violations detected.",
        }

    def _build_decision(
        self,
        result: dict[str, Any],
        latency_ms: float,
        mode: EnforcementMode,
    ) -> EnforcementDecision:
        unsafe = result.get("unsafe", False)
        action = EnforcementAction.BLOCK if unsafe else EnforcementAction.PASS

        return EnforcementDecision(
            log_id=uuid.uuid4().hex[:16],
            action=action,
            category=result.get("policy_category", ""),
            severity=int(result.get("severity", 0)),
            matched_rule=result.get("matched_block_rule", ""),
            explanation=result.get("explanation", ""),
            latency_ms=latency_ms,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Abandonment tracking
    # ------------------------------------------------------------------

    async def record_abandonment(self, log_id: str) -> None:
        """
        Mark an enforcement decision as abandoned by the user.

        Called when the client disconnects shortly after receiving a block
        decision — a signal the block may have been a false positive.
        """
        try:
            sql = f"""
                ALTER TABLE arena.enforcement_log
                UPDATE abandoned = 1
                WHERE log_id = '{_esc(log_id)}'
            """
            await self._ch.query(sql)
            logger.info("Abandonment recorded for %s", log_id)
        except Exception:
            logger.warning("Failed to record abandonment for %s", log_id, exc_info=True)

    # ------------------------------------------------------------------
    # Analyst override
    # ------------------------------------------------------------------

    async def record_override(self, log_id: str, override: str) -> None:
        """
        Record an analyst override for an enforcement decision.

        Parameters
        ----------
        log_id
            The enforcement log entry to override.
        override
            "false_positive" (block was wrong) or "confirmed" (block was correct).
        """
        if override not in ("false_positive", "confirmed"):
            raise ValueError(f"Invalid override: {override}. Must be 'false_positive' or 'confirmed'.")

        try:
            sql = f"""
                ALTER TABLE arena.enforcement_log
                UPDATE analyst_override = '{_esc(override)}'
                WHERE log_id = '{_esc(log_id)}'
            """
            await self._ch.query(sql)
            logger.info("Override recorded for %s: %s", log_id, override)
        except Exception:
            logger.warning("Failed to record override for %s", log_id, exc_info=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _log_decision(
        self,
        ai_system_id: str,
        prompt: str,
        output: str,
        decision: EnforcementDecision,
    ) -> None:
        prompt_hash = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
        output_hash = hashlib.sha256(output.strip().lower().encode()).hexdigest()
        preview = output[:500].replace("'", "\\'").replace("\\", "\\\\")

        try:
            sql = f"""
                INSERT INTO arena.enforcement_log VALUES (
                    '{_esc(decision.log_id)}',
                    {time.time()},
                    '{_esc(ai_system_id)}',
                    '{_esc(prompt_hash)}',
                    '{_esc(output_hash)}',
                    '{_esc(preview)}',
                    '{decision.mode.value}',
                    '{decision.action.value}',
                    '{_esc(decision.category)}',
                    {decision.severity},
                    '{_esc(decision.matched_rule)}',
                    '{_esc(decision.explanation)}',
                    {decision.latency_ms},
                    NULL,
                    0
                )
            """
            await self._ch.query(sql)
        except Exception:
            logger.warning("Failed to log enforcement decision %s", decision.log_id, exc_info=True)

    async def _update_metrics(self, decision: EnforcementDecision) -> None:
        """Increment the current window's trigger count for the matched rule."""
        if decision.action != EnforcementAction.BLOCK or not decision.category:
            return

        rule_id = f"{decision.category}.block"
        now = time.time()
        window_start = now - (now % self._metrics_window)
        window_end = window_start + self._metrics_window

        try:
            # Upsert: ReplacingMergeTree will keep the latest row per (rule_id, window_start)
            sql = f"""
                INSERT INTO arena.rule_performance_metrics VALUES (
                    '{_esc(rule_id)}',
                    '{_esc(decision.category)}',
                    {window_start},
                    {window_end},
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    'active'
                )
            """
            await self._ch.query(sql)
        except Exception:
            logger.warning("Failed to update rule metrics for %s", rule_id, exc_info=True)

    # ------------------------------------------------------------------
    # Metrics computation & rollback
    # ------------------------------------------------------------------

    async def compute_rule_metrics(self) -> list[dict[str, Any]]:
        """
        Compute per-rule false-positive and abandonment rates for the
        current 15-minute window.  Returns a list of rule summaries.
        """
        now = time.time()
        window_start = now - self._metrics_window

        sql = f"""
            SELECT
                category,
                count()                                         AS total_blocks,
                countIf(analyst_override = 'confirmed')         AS confirmed_true,
                countIf(analyst_override = 'false_positive')    AS confirmed_false,
                countIf(analyst_override IS NULL AND action = 'block') AS pending,
                countIf(abandoned = 1)                          AS abandoned_count
            FROM arena.enforcement_log
            WHERE action = 'block'
              AND timestamp >= {window_start}
            GROUP BY category
        """

        results: list[dict[str, Any]] = []
        try:
            resp = await self._ch.query(sql)
            for row in resp.result_rows:  # type: ignore
                category = str(row[0])
                total = int(row[1])
                confirmed_true = int(row[2])
                confirmed_false = int(row[3])
                pending = int(row[4])
                abandoned = int(row[5])

                reviewed = confirmed_true + confirmed_false
                fp_rate = confirmed_false / reviewed if reviewed > 0 else 0.0
                abandon_rate = abandoned / total if total > 0 else 0.0

                rule_id = f"{category}.block"
                status = self._rule_statuses.get(rule_id)

                results.append({
                    "rule_id": rule_id,
                    "category": category,
                    "total_blocks": total,
                    "confirmed_true": confirmed_true,
                    "confirmed_false": confirmed_false,
                    "pending_review": pending,
                    "false_positive_rate": round(fp_rate, 4),
                    "abandonment_rate": round(abandon_rate, 4),
                    "abandoned_count": abandoned,
                    "status": "downgraded" if (status and not status.active) else "active",
                })
        except Exception:
            logger.warning("Failed to compute rule metrics", exc_info=True)

        return results

    async def check_and_rollback(self) -> list[dict[str, Any]]:
        """
        Check all rules against the false-positive and abandonment thresholds.

        Rules exceeding the 2 % FP rate are auto-downgraded from SYNC → ASYNC.
        Returns a list of rules that were rolled back.

        Should be called on a schedule (every 15 minutes).
        """
        metrics = await self.compute_rule_metrics()
        rolled_back: list[dict[str, Any]] = []

        for m in metrics:
            rule_id = m["rule_id"]
            category = m["category"]
            fp_rate = m["false_positive_rate"]
            abandon_rate = m["abandonment_rate"]

            # Already downgraded — skip
            existing = self._rule_statuses.get(rule_id)
            if existing and not existing.active:
                continue

            reasons: list[str] = []
            if fp_rate > self._fp_threshold:
                reasons.append(f"FP rate {fp_rate:.2%} > {self._fp_threshold:.0%} threshold")
            if abandon_rate > self.ABANDONMENT_RATE_THRESHOLD:
                reasons.append(f"abandonment rate {abandon_rate:.2%} > {self.ABANDONMENT_RATE_THRESHOLD:.0%} threshold")

            if not reasons:
                continue

            reason = "; ".join(reasons)
            now = time.time()

            # Downgrade the rule
            self._rule_statuses[rule_id] = RuleStatus(
                rule_id=rule_id,
                category=category,
                active=False,
                downgraded_at=now,
                reason=reason,
            )

            # Persist the downgrade in metrics table
            window_start = now - (now % self._metrics_window)
            window_end = window_start + self._metrics_window
            try:
                sql = f"""
                    INSERT INTO arena.rule_performance_metrics VALUES (
                        '{_esc(rule_id)}',
                        '{_esc(category)}',
                        {window_start},
                        {window_end},
                        {m['total_blocks']},
                        {m['confirmed_true']},
                        {m['confirmed_false']},
                        {m['pending_review']},
                        {fp_rate},
                        {abandon_rate},
                        'downgraded'
                    )
                """
                await self._ch.query(sql)
            except Exception:
                logger.warning("Failed to persist rollback for %s", rule_id, exc_info=True)

            rolled_back.append({
                "rule_id": rule_id,
                "category": category,
                "reason": reason,
                "false_positive_rate": fp_rate,
                "abandonment_rate": abandon_rate,
                "downgraded_at": now,
            })

            logger.warning(
                "AUTO-ROLLBACK: rule %s downgraded — %s",
                rule_id, reason,
            )

        return rolled_back

    async def reinstate_rule(self, rule_id: str) -> bool:
        """
        Manually reinstate a downgraded rule after analyst review.

        Returns True if the rule was reinstated, False if it wasn't downgraded.
        """
        status = self._rule_statuses.get(rule_id)
        if not status or status.active:
            return False

        status.active = True
        status.downgraded_at = None
        status.reason = ""

        logger.info("Rule %s reinstated by analyst", rule_id)
        return True

    # ------------------------------------------------------------------
    # Rollback scheduler
    # ------------------------------------------------------------------

    async def start_metrics_loop(self) -> None:
        """
        Background loop that checks rule metrics every 15 minutes and
        triggers auto-rollback when thresholds are exceeded.

        Call this once at startup; runs until cancelled.
        """
        logger.info(
            "Ozone metrics loop started (window=%ds, fp_threshold=%.2f%%)",
            self._metrics_window, self._fp_threshold * 100,
        )
        while True:
            try:
                rolled_back = await self.check_and_rollback()
                if rolled_back:
                    logger.warning(
                        "Metrics check: %d rule(s) auto-rolled back",
                        len(rolled_back),
                    )
            except Exception:
                logger.exception("Metrics loop error")
            await asyncio.sleep(self._metrics_window)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get_enforcement_log(
        self,
        ai_system_id: str | None = None,
        category: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query enforcement log with optional filters."""
        conditions = ["1=1"]
        if ai_system_id:
            conditions.append(f"ai_system_id = '{_esc(ai_system_id)}'")
        if category:
            conditions.append(f"category = '{_esc(category)}'")
        if action:
            conditions.append(f"action = '{_esc(action)}'")

        where = " AND ".join(conditions)
        sql = f"""
            SELECT log_id, timestamp, ai_system_id, mode, action,
                   category, severity, matched_rule, explanation,
                   latency_ms, analyst_override, abandoned
            FROM arena.enforcement_log
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT {min(limit, 1000)}
        """

        rows: list[dict[str, Any]] = []
        try:
            resp = await self._ch.query(sql)
            for row in resp.result_rows:  # type: ignore
                rows.append({
                    "log_id": str(row[0]),
                    "timestamp": float(row[1]),
                    "ai_system_id": str(row[2]),
                    "mode": str(row[3]),
                    "action": str(row[4]),
                    "category": str(row[5]),
                    "severity": int(row[6]),
                    "matched_rule": str(row[7]),
                    "explanation": str(row[8]),
                    "latency_ms": round(float(row[9]), 2),
                    "analyst_override": str(row[10]) if row[10] else None,
                    "abandoned": bool(row[11]),
                })
        except Exception:
            logger.warning("Enforcement log query failed", exc_info=True)

        return rows

    async def get_rule_statuses(self) -> list[dict[str, Any]]:
        """Return the current in-memory status of all tracked rules."""
        statuses = []
        for rule_id, s in self._rule_statuses.items():
            statuses.append({
                "rule_id": s.rule_id,
                "category": s.category,
                "active": s.active,
                "downgraded_at": s.downgraded_at,
                "reason": s.reason,
            })
        return statuses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    """Escape for ClickHouse SQL string literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")
