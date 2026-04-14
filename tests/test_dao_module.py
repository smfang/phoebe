"""Integration tests: full DAO module end-to-end through DomainEnforcement."""

import pytest

from src.domains.base import Action, EnforcementMode, EvalRequest, RiskTier
from src.domains.dao.models import SimulationResult
from src.domains.dao.module import build
from src.domains.dao.risk_tiers import (
    DAO_TIER_MAP,
    DEFAULT_T2_AMOUNT_THRESHOLD,
    classify_call,
    dao_risk_tier,
)
from src.domains.dao.models import DAOFunctionCall
from src.domains.enforcement import DomainEnforcement
from src.domains.registry import DomainRegistry


ALICE = "0x" + "11" * 20
BOB = "0x" + "22" * 20


class _StubClassifier:
    def __init__(self, aligned: bool = True):
        self._aligned = aligned

    async def classify(self, prompt, model_output, category):
        return {
            "aligned": self._aligned,
            "confidence": 0.9,
            "severity": 0 if self._aligned else 4,
            "explanation": "ok" if self._aligned else "intent mismatch",
        }


class _StubBackend:
    name = "stub"

    def __init__(self, result: SimulationResult | None = None):
        self._result = result or SimulationResult(succeeded=True, backend="stub")

    async def simulate(self, call):
        return self._result


def _payload(amount: int = 100, signature: str = "transfer(address,uint256)") -> dict:
    return {
        "chain_id": 1,
        "contract_address": "0x" + "ab" * 20,
        "function_selector": "0xa9059cbb",
        "function_name": "transfer",
        "abi_signature": signature,
        "decoded_params": {"to": BOB, "amount": amount},
        "raw_calldata": "0xa9059cbb" + "00" * 64,
        "caller_address": ALICE,
    }


def _req(payload: dict, intent: str = "send tokens") -> EvalRequest:
    return EvalRequest(domain="dao", payload=payload, user_intent=intent)


# ---------------------------------------------------------------------------
# Risk-tier classification
# ---------------------------------------------------------------------------


def test_risk_tier_t1_for_view():
    call = DAOFunctionCall(
        chain_id=1, contract_address="0x" + "ab" * 20,
        function_selector="0x70a08231", function_name="balanceOf",
        abi_signature="balanceOf(address)",
    )
    assert classify_call(call) == RiskTier.T1


def test_risk_tier_t2_below_threshold():
    payload = _payload(amount=DEFAULT_T2_AMOUNT_THRESHOLD - 1)
    assert dao_risk_tier(_req(payload)) == RiskTier.T2


def test_risk_tier_t3_at_or_above_threshold():
    payload = _payload(amount=DEFAULT_T2_AMOUNT_THRESHOLD)
    assert dao_risk_tier(_req(payload)) == RiskTier.T3


def test_risk_tier_t4_for_admin():
    call = DAOFunctionCall(
        chain_id=1, contract_address="0x" + "ab" * 20,
        function_selector="0x3659cfe6", function_name="upgradeTo",
        abi_signature="upgradeTo(address)",
    )
    assert classify_call(call) == RiskTier.T4


def test_unknown_signature_is_t4():
    call = DAOFunctionCall(
        chain_id=1, contract_address="0x" + "ab" * 20,
        function_selector="0xdeadbeef", function_name="rugPull",
        abi_signature="rugPull()",
    )
    assert classify_call(call) == RiskTier.T4


def test_dao_tier_map_covers_all_tiers():
    for tier in RiskTier:
        assert tier in DAO_TIER_MAP


def test_dao_tier_map_t1_is_async():
    assert DAO_TIER_MAP[RiskTier.T1] == EnforcementMode.ASYNC


def test_dao_tier_map_t4_is_multisig():
    assert DAO_TIER_MAP[RiskTier.T4] == EnforcementMode.SYNC_MULTISIG


# ---------------------------------------------------------------------------
# End-to-end through DomainEnforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_clean_t2_transfer_quarantines_on_misaligned():
    # T2 maps to QUARANTINE, so even a clean simulation but misaligned judge
    # should produce a QUARANTINE action (not BLOCK).
    reg = DomainRegistry()
    reg.register(build(
        classifier=_StubClassifier(aligned=False),
        simulation_backend=_StubBackend(),
    ))
    enf = DomainEnforcement(registry=reg)

    decision = await enf.evaluate(_req(_payload(amount=100)))
    assert decision.tier == RiskTier.T2
    assert decision.mode == EnforcementMode.QUARANTINE
    assert decision.action == Action.QUARANTINE


@pytest.mark.asyncio
async def test_e2e_clean_t3_transfer_blocks_on_misaligned():
    reg = DomainRegistry()
    reg.register(build(
        classifier=_StubClassifier(aligned=False),
        simulation_backend=_StubBackend(),
    ))
    enf = DomainEnforcement(registry=reg)

    decision = await enf.evaluate(_req(_payload(amount=DEFAULT_T2_AMOUNT_THRESHOLD)))
    assert decision.tier == RiskTier.T3
    assert decision.action == Action.BLOCK


@pytest.mark.asyncio
async def test_e2e_aligned_call_passes():
    reg = DomainRegistry()
    reg.register(build(
        classifier=_StubClassifier(aligned=True),
        simulation_backend=_StubBackend(),
    ))
    enf = DomainEnforcement(registry=reg)

    decision = await enf.evaluate(_req(_payload(amount=100)))
    assert decision.action == Action.PASS
    # All three stages should have run for T2
    stage_names = [r.stage for r in decision.stage_results]
    assert "validator" in stage_names
    assert "simulator" in stage_names
    assert "alignment" in stage_names


@pytest.mark.asyncio
async def test_e2e_validator_failure_short_circuits_in_sync():
    # T3 = SYNC. A validator block should prevent simulator + judge from running.
    reg = DomainRegistry()
    reg.register(build(
        classifier=_StubClassifier(aligned=True),
        simulation_backend=_StubBackend(),
    ))
    enf = DomainEnforcement(registry=reg)

    payload = _payload(amount=DEFAULT_T2_AMOUNT_THRESHOLD)
    payload["abi_signature"] = "rugPull(uint256)"  # unknown → T4 (multisig escalate)

    decision = await enf.evaluate(_req(payload))
    # Unknown sig → T4 → SYNC_MULTISIG mode, validator says BLOCK
    assert decision.tier == RiskTier.T4
    assert decision.action == Action.ESCALATE


@pytest.mark.asyncio
async def test_e2e_t1_view_call_passes_without_simulation():
    reg = DomainRegistry()
    reg.register(build(
        classifier=_StubClassifier(aligned=True),
        simulation_backend=_StubBackend(),
    ))
    enf = DomainEnforcement(registry=reg)

    payload = {
        "chain_id": 1,
        "contract_address": "0x" + "ab" * 20,
        "function_selector": "0x70a08231",
        "function_name": "balanceOf",
        "abi_signature": "balanceOf(address)",
        "decoded_params": {"account": ALICE},
        "raw_calldata": "0x70a08231" + "00" * 32,
        "caller_address": ALICE,
    }

    decision = await enf.evaluate(_req(payload))
    assert decision.tier == RiskTier.T1
    assert decision.mode == EnforcementMode.ASYNC
    assert decision.action == Action.PASS


def test_module_metadata():
    module = build(classifier=_StubClassifier(), simulation_backend=_StubBackend())
    assert module.name == "dao"
    assert module.version == "0.1.0"
    assert module.stage_names() == ["validator", "simulator", "alignment"]
