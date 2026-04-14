"""Tests for the DomainEnforcement orchestrator."""

from typing import Any

import pytest

from src.domains.base import (
    Action,
    DomainModule,
    EnforcementMode,
    EvalRequest,
    EvalState,
    RiskTier,
    StageResult,
)
from src.domains.enforcement import DomainEnforcement
from src.domains.registry import DomainRegistry


class _PassStage:
    name = "pass_stage"

    async def run(self, req: EvalRequest, state: EvalState) -> StageResult:
        return StageResult(stage=self.name, action=Action.PASS, reason="ok")


class _BlockStage:
    name = "block_stage"

    async def run(self, req: EvalRequest, state: EvalState) -> StageResult:
        return StageResult(
            stage=self.name, action=Action.BLOCK, severity=4, reason="nope"
        )


class _RaisingStage:
    name = "raise_stage"

    async def run(self, req: EvalRequest, state: EvalState) -> StageResult:
        raise RuntimeError("boom")


def _module(
    stages: list[Any],
    tier: RiskTier = RiskTier.T3,
    mode: EnforcementMode = EnforcementMode.SYNC,
    name: str = "test",
) -> DomainModule:
    return DomainModule(
        name=name,
        version="0.1.0",
        description="test",
        stages=stages,
        risk_tier_fn=lambda req: tier,
        tier_to_mode={tier: mode},
    )


def _request(domain: str = "test", payload: dict | None = None) -> EvalRequest:
    return EvalRequest(domain=domain, payload=payload or {})


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_domain_blocks_with_t4():
    reg = DomainRegistry()
    enf = DomainEnforcement(registry=reg)
    decision = await enf.evaluate(_request("does_not_exist"))
    assert decision.action == Action.BLOCK
    assert decision.tier == RiskTier.T4
    assert "unknown domain" in decision.reason


@pytest.mark.asyncio
async def test_all_pass_returns_pass():
    reg = DomainRegistry()
    reg.register(_module([_PassStage(), _PassStage()]))
    enf = DomainEnforcement(registry=reg)
    decision = await enf.evaluate(_request())
    assert decision.action == Action.PASS
    assert len(decision.stage_results) == 2


@pytest.mark.asyncio
async def test_sync_short_circuits_on_block():
    reg = DomainRegistry()
    reg.register(_module([_BlockStage(), _PassStage()], mode=EnforcementMode.SYNC))
    enf = DomainEnforcement(registry=reg)
    decision = await enf.evaluate(_request())
    assert decision.action == Action.BLOCK
    # SYNC short-circuits, so the second stage never runs
    assert len(decision.stage_results) == 1


@pytest.mark.asyncio
async def test_async_mode_logs_but_does_not_block():
    reg = DomainRegistry()
    reg.register(_module([_BlockStage()], tier=RiskTier.T1, mode=EnforcementMode.ASYNC))
    enf = DomainEnforcement(registry=reg)
    decision = await enf.evaluate(_request())
    assert decision.action == Action.PASS
    assert "would block" in decision.reason


@pytest.mark.asyncio
async def test_quarantine_mode_quarantines():
    reg = DomainRegistry()
    reg.register(
        _module([_BlockStage()], tier=RiskTier.T2, mode=EnforcementMode.QUARANTINE)
    )
    enf = DomainEnforcement(registry=reg)
    decision = await enf.evaluate(_request())
    assert decision.action == Action.QUARANTINE


@pytest.mark.asyncio
async def test_multisig_mode_escalates():
    reg = DomainRegistry()
    reg.register(
        _module([_BlockStage()], tier=RiskTier.T4, mode=EnforcementMode.SYNC_MULTISIG)
    )
    enf = DomainEnforcement(registry=reg)
    decision = await enf.evaluate(_request())
    assert decision.action == Action.ESCALATE
    assert "multisig" in decision.reason


@pytest.mark.asyncio
async def test_stage_exception_is_caught_and_blocks_in_sync():
    reg = DomainRegistry()
    reg.register(_module([_RaisingStage()], mode=EnforcementMode.SYNC))
    enf = DomainEnforcement(registry=reg)
    decision = await enf.evaluate(_request())
    assert decision.action == Action.BLOCK
    assert "stage_error" in decision.stage_results[0].reason


@pytest.mark.asyncio
async def test_decision_carries_module_metadata():
    reg = DomainRegistry()
    reg.register(_module([_PassStage()], name="dao"))
    enf = DomainEnforcement(registry=reg)
    decision = await enf.evaluate(_request("dao"))
    assert decision.domain == "dao"
    assert decision.module_version == "0.1.0"
    assert decision.tier == RiskTier.T3


@pytest.mark.asyncio
async def test_persistence_called_when_store_present():
    reg = DomainRegistry()
    reg.register(_module([_PassStage()]))

    seen_sql: list[str] = []

    class _Store:
        async def query(self, sql: str):
            seen_sql.append(sql)
            return None

    enf = DomainEnforcement(registry=reg, store=_Store())
    await enf.evaluate(_request())
    assert len(seen_sql) == 1
    assert "INSERT INTO arena.domain_evaluations" in seen_sql[0]


@pytest.mark.asyncio
async def test_ozone_delegation_called_when_present():
    reg = DomainRegistry()
    reg.register(_module([_PassStage()]))

    seen: dict[str, Any] = {}

    class _Ozone:
        async def evaluate(self, content, category, mode_override=None, external_signal=None):
            seen["category"] = category
            seen["mode_override"] = mode_override
            return None

    enf = DomainEnforcement(registry=reg, ozone=_Ozone())
    await enf.evaluate(_request())
    assert seen["category"] == "test_T3"
    assert seen["mode_override"] == EnforcementMode.SYNC


@pytest.mark.asyncio
async def test_ozone_failure_does_not_break_decision():
    reg = DomainRegistry()
    reg.register(_module([_PassStage()]))

    class _BrokenOzone:
        async def evaluate(self, **kwargs):
            raise RuntimeError("ozone offline")

    enf = DomainEnforcement(registry=reg, ozone=_BrokenOzone())
    decision = await enf.evaluate(_request())
    assert decision.action == Action.PASS
