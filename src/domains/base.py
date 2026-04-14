"""
Core types for the pluggable domain enforcement framework.

A domain module bundles three things:
    1. A list of stages that examine an `EvalRequest` in order
    2. A function that derives a `RiskTier` from the request
    3. A mapping from risk tier to `EnforcementMode`

The orchestrator (`DomainEnforcement`) is domain-agnostic — it just runs
stages, threads state through them, and maps the resulting tier to a mode.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums shared across all domains
# ---------------------------------------------------------------------------


class RiskTier(str, Enum):
    """Risk classification for any domain operation.

    Domains map their own risk semantics onto these four tiers so that
    cross-cutting infrastructure (Ozone, dashboard, billing) can treat
    risk uniformly.
    """

    T1 = "T1"  # informational / read-only
    T2 = "T2"  # low-impact action below threshold
    T3 = "T3"  # significant action above threshold
    T4 = "T4"  # privileged / admin / irreversible


class EnforcementMode(str, Enum):
    """How the system reacts when a stage flags risk.

    Mirrors the modes the Ozone enforcement layer uses, so a domain
    decision can be delegated to Ozone with no translation.
    """

    ASYNC = "ASYNC"  # log-only, never block
    QUARANTINE = "QUARANTINE"  # block + queue for human review
    SYNC = "SYNC"  # block inline if any stage says BLOCK
    SYNC_MULTISIG = "SYNC_MULTISIG"  # block + require N-of-M signers + timelock


class Action(str, Enum):
    """Action a single stage recommends, or the final decision."""

    PASS = "PASS"
    BLOCK = "BLOCK"
    QUARANTINE = "QUARANTINE"
    ESCALATE = "ESCALATE"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class EvalRequest(BaseModel):
    """Opaque request envelope passed to a domain module.

    The `payload` is a domain-specific dict; each domain's first stage
    parses it into a typed model. Keeping the boundary opaque means
    third-party domain modules can be installed without modifying the
    orchestrator's types.
    """

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    domain: str
    content_type: str = "application/json"
    payload: dict[str, Any] = Field(default_factory=dict)
    user_intent: str = ""
    caller: str = ""  # wallet, user id, agent name — domain-defined
    submitted_at: float = Field(default_factory=time.time)


class StageResult(BaseModel):
    """Result emitted by a single stage."""

    stage: str
    action: Action = Action.PASS
    severity: int = 0  # 0 = none, 1–5 = low → catastrophic
    confidence: float = 1.0
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float = 0.0


class EvalState(BaseModel):
    """Accumulator threaded through all stages.

    Stages can read prior results from `results` and write to `scratch`
    for downstream stages.
    """

    request_id: str
    domain: str
    module_version: str
    tier: RiskTier
    results: list[StageResult] = Field(default_factory=list)
    scratch: dict[str, Any] = Field(default_factory=dict)

    def append(self, result: StageResult) -> None:
        self.results.append(result)

    def has_block(self) -> bool:
        return any(r.action == Action.BLOCK for r in self.results)

    def max_severity(self) -> int:
        return max((r.severity for r in self.results), default=0)


class EnforcementDecision(BaseModel):
    """Final verdict produced by `DomainEnforcement.evaluate()`."""

    request_id: str
    domain: str
    module_version: str
    tier: RiskTier
    mode: EnforcementMode
    action: Action
    reason: str = ""
    severity: int = 0
    stage_results: list[StageResult] = Field(default_factory=list)
    decided_at: float = Field(default_factory=time.time)
    elapsed_ms: float = 0.0

    def to_log_row(self) -> dict[str, Any]:
        """Flat dict suitable for ClickHouse insertion."""
        return {
            "request_id": self.request_id,
            "domain": self.domain,
            "module_version": self.module_version,
            "tier": self.tier.value,
            "mode": self.mode.value,
            "action": self.action.value,
            "severity": self.severity,
            "reason": self.reason,
            "stage_results": self.model_dump_json(include={"stage_results"}),
            "decided_at": self.decided_at,
            "elapsed_ms": self.elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Stage and module protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Stage(Protocol):
    """A single verification step within a domain module.

    Implementations should be small, composable, and side-effect-free
    aside from network/disk I/O they declare. They MUST be safe to call
    concurrently across requests.
    """

    name: str

    async def run(self, req: EvalRequest, state: EvalState) -> StageResult: ...


# A risk-tier function takes a request and returns the tier classification.
RiskTierFn = Callable[[EvalRequest], RiskTier]

# A health-check function returns a status dict; called by /api/domains/healthcheck.
HealthcheckFn = Callable[[], Awaitable[dict[str, Any]]]


class DomainModule(BaseModel):
    """A pluggable domain module — what a third-party package registers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    version: str
    description: str = ""
    stages: list[Any]  # list[Stage] — Pydantic can't validate Protocol
    risk_tier_fn: RiskTierFn
    tier_to_mode: dict[RiskTier, EnforcementMode]
    healthcheck_fn: HealthcheckFn | None = None

    def stage_names(self) -> list[str]:
        return [getattr(s, "name", s.__class__.__name__) for s in self.stages]
