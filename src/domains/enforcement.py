"""
Domain-agnostic enforcement orchestrator.

`DomainEnforcement.evaluate()` runs a registered domain module's stages in
order, threads `EvalState` through them, applies tier→mode mapping, and
optionally delegates to the Ozone enforcement layer for the final action.

Stays decoupled from Ozone: if no Ozone instance is provided, the
orchestrator computes the action locally based on stage results and the
mapped mode.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from src.domains.base import (
    Action,
    EnforcementDecision,
    EnforcementMode,
    EvalRequest,
    EvalState,
    StageResult,
)
from src.domains.registry import DomainRegistry, UnknownDomainError

logger = logging.getLogger(__name__)


class _OzoneLike(Protocol):
    """Soft dependency on the Ozone interface (Phase 1).

    Anything that exposes this shape can be plugged in. The orchestrator
    only calls `evaluate(...)` and ignores the rest.
    """

    async def evaluate(
        self,
        content: str,
        category: str,
        mode_override: Any = None,
        external_signal: dict[str, Any] | None = None,
    ) -> Any: ...


class _StoreLike(Protocol):
    """Persistence interface — needs only an async query() method."""

    async def query(self, sql: str) -> Any: ...


class DomainEnforcement:
    """Orchestrates a domain module's stages and emits an `EnforcementDecision`."""

    def __init__(
        self,
        registry: DomainRegistry,
        ozone: _OzoneLike | None = None,
        store: _StoreLike | None = None,
    ) -> None:
        self._registry = registry
        self._ozone = ozone
        self._store = store

    async def evaluate(self, req: EvalRequest) -> EnforcementDecision:
        """Run all stages for the requested domain, return final decision."""
        started = time.perf_counter()

        try:
            module = self._registry.get(req.domain)
        except UnknownDomainError:
            return self._unknown_domain_decision(req, started)

        tier = module.risk_tier_fn(req)
        mode = module.tier_to_mode.get(tier, EnforcementMode.SYNC)

        state = EvalState(
            request_id=req.request_id,
            domain=req.domain,
            module_version=module.version,
            tier=tier,
        )

        for stage in module.stages:
            stage_started = time.perf_counter()
            try:
                result = await stage.run(req, state)
            except Exception as exc:
                logger.exception("Stage %s raised", getattr(stage, "name", stage))
                result = StageResult(
                    stage=getattr(stage, "name", stage.__class__.__name__),
                    action=Action.BLOCK if mode == EnforcementMode.SYNC else Action.ESCALATE,
                    severity=3,
                    reason=f"stage_error: {type(exc).__name__}",
                    details={"error": str(exc)},
                )
            result.elapsed_ms = (time.perf_counter() - stage_started) * 1000.0
            state.append(result)

            # SYNC mode: short-circuit on first BLOCK to save downstream work
            if mode == EnforcementMode.SYNC and result.action == Action.BLOCK:
                break

        action, reason = self._resolve_action(state, mode)
        decision = EnforcementDecision(
            request_id=req.request_id,
            domain=req.domain,
            module_version=module.version,
            tier=tier,
            mode=mode,
            action=action,
            reason=reason,
            severity=state.max_severity(),
            stage_results=state.results,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

        # Optionally delegate to Ozone for cross-domain enforcement effects
        # (rate-of-block tracking, automated rollback, dashboard feed).
        if self._ozone is not None:
            try:
                await self._ozone.evaluate(
                    content=req.content_type,
                    category=f"{req.domain}_{tier.value}",
                    mode_override=mode,
                    external_signal=decision.model_dump(),
                )
            except Exception:
                logger.warning("Ozone delegation failed for %s", req.request_id, exc_info=True)

        await self._persist(decision)
        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_action(self, state: EvalState, mode: EnforcementMode) -> tuple[Action, str]:
        """Compute final action from stage results + enforcement mode."""
        blocking = [r for r in state.results if r.action == Action.BLOCK]
        if not blocking:
            return Action.PASS, "all stages passed"

        # Pick the highest-severity block reason for explanation
        worst = max(blocking, key=lambda r: r.severity)
        reason = f"{worst.stage}: {worst.reason}" if worst.reason else worst.stage

        if mode == EnforcementMode.ASYNC:
            # Log-only, never block — tier said this is informational
            return Action.PASS, f"asyncronous flag (would block: {reason})"
        if mode == EnforcementMode.QUARANTINE:
            return Action.QUARANTINE, reason
        if mode == EnforcementMode.SYNC_MULTISIG:
            return Action.ESCALATE, f"multisig required: {reason}"
        # SYNC
        return Action.BLOCK, reason

    def _unknown_domain_decision(
        self, req: EvalRequest, started: float
    ) -> EnforcementDecision:
        from src.domains.base import RiskTier

        return EnforcementDecision(
            request_id=req.request_id,
            domain=req.domain,
            module_version="unknown",
            tier=RiskTier.T4,
            mode=EnforcementMode.SYNC,
            action=Action.BLOCK,
            reason=f"unknown domain: {req.domain}",
            severity=4,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _persist(self, decision: EnforcementDecision) -> None:
        """Insert the decision into ClickHouse if a store is wired."""
        if self._store is None:
            return
        row = decision.to_log_row()
        # Escape strings safely for direct SQL — ClickHouse client doesn't
        # parameterize on the query() call we use here.
        sql = f"""
            INSERT INTO arena.domain_evaluations (
                request_id, domain, module_version, tier, mode, action,
                severity, reason, stage_results, decided_at, elapsed_ms
            ) VALUES (
                '{_esc(row["request_id"])}',
                '{_esc(row["domain"])}',
                '{_esc(row["module_version"])}',
                '{_esc(row["tier"])}',
                '{_esc(row["mode"])}',
                '{_esc(row["action"])}',
                {int(row["severity"])},
                '{_esc(row["reason"])}',
                '{_esc(row["stage_results"])}',
                {float(row["decided_at"])},
                {float(row["elapsed_ms"])}
            )
        """
        try:
            await self._store.query(sql)
        except Exception:
            logger.warning("Failed to persist domain decision", exc_info=True)


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("'", "\\'")
