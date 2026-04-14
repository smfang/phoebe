"""Tests for the DAO SimulationRunner stage."""

import pytest

from src.domains.base import Action, EvalRequest, EvalState, RiskTier
from src.domains.dao.models import (
    DAOFunctionCall,
    ERC20Transfer,
    OwnershipChange,
    SimulationResult,
)
from src.domains.dao.simulator import (
    SimulationRunner,
    _BackendUnreachable,
    _post_check,
)

ALICE = "0x" + "11" * 20
BOB = "0x" + "22" * 20


def _state(tier: RiskTier = RiskTier.T2) -> EvalState:
    s = EvalState(request_id="r1", domain="dao", module_version="0.1.0", tier=tier)
    s.scratch["call"] = {
        "chain_id": 1,
        "contract_address": "0x" + "ab" * 20,
        "function_selector": "0xa9059cbb",
        "function_name": "transfer",
        "abi_signature": "transfer(address,uint256)",
        "decoded_params": {"to": BOB, "amount": 100},
        "raw_calldata": "0xa9059cbb",
        "caller_address": ALICE,
    }
    return s


def _req() -> EvalRequest:
    return EvalRequest(domain="dao", payload={})


class _StubBackend:
    name = "stub"

    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises

    async def simulate(self, call):
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.mark.asyncio
async def test_t1_skips_simulation():
    sim = SimulationRunner(backend=_StubBackend())
    result = await sim.run(_req(), _state(tier=RiskTier.T1))
    assert result.action == Action.PASS
    assert "skipped" in result.reason


@pytest.mark.asyncio
async def test_clean_simulation_passes():
    backend = _StubBackend(result=SimulationResult(succeeded=True, gas_used=21000, backend="stub"))
    sim = SimulationRunner(backend=backend)
    result = await sim.run(_req(), _state())
    assert result.action == Action.PASS
    assert "clean" in result.reason


@pytest.mark.asyncio
async def test_reverted_simulation_blocks():
    backend = _StubBackend(result=SimulationResult(
        succeeded=False, revert_reason="ERC20: insufficient balance", backend="stub"
    ))
    sim = SimulationRunner(backend=backend)
    result = await sim.run(_req(), _state())
    assert result.action == Action.BLOCK
    assert "reverted" in result.reason


@pytest.mark.asyncio
async def test_backend_unreachable_t3_fail_closed():
    backend = _StubBackend(raises=_BackendUnreachable("anvil down"))
    sim = SimulationRunner(backend=backend)
    result = await sim.run(_req(), _state(tier=RiskTier.T3))
    assert result.action == Action.BLOCK
    assert "fail-closed" in result.reason


@pytest.mark.asyncio
async def test_backend_unreachable_t2_degrades():
    backend = _StubBackend(raises=_BackendUnreachable("anvil down"))
    sim = SimulationRunner(backend=backend)
    result = await sim.run(_req(), _state(tier=RiskTier.T2))
    assert result.action == Action.PASS
    assert "degraded" in result.reason


@pytest.mark.asyncio
async def test_suspicious_flags_block_t3():
    backend = _StubBackend(result=SimulationResult(
        succeeded=True,
        backend="stub",
        ownership_changes=[OwnershipChange(contract="0x" + "ab" * 20, previous_owner=ALICE, new_owner=BOB)],
        suspicious_flags=["ownership transferred"],
    ))
    sim = SimulationRunner(backend=backend)
    result = await sim.run(_req(), _state(tier=RiskTier.T3))
    assert result.action == Action.BLOCK
    assert "ownership" in result.reason


@pytest.mark.asyncio
async def test_state_scratch_populated_for_downstream():
    backend = _StubBackend(result=SimulationResult(succeeded=True, backend="stub"))
    sim = SimulationRunner(backend=backend)
    state = _state()
    await sim.run(_req(), state)
    assert "simulation" in state.scratch


def test_post_check_self_dealing_flag():
    call = DAOFunctionCall(
        chain_id=1,
        contract_address="0x" + "ab" * 20,
        function_selector="0xa9059cbb",
        function_name="transfer",
        abi_signature="transfer(address,uint256)",
        caller_address=ALICE,
    )
    sim = SimulationResult(
        succeeded=True,
        backend="stub",
        erc20_transfers=[ERC20Transfer(token="0x" + "ab" * 20, from_addr=BOB, to_addr=ALICE, amount=100)],
    )
    flags = _post_check(call, sim)
    assert any("self-dealing" in f for f in flags)


def test_post_check_no_self_dealing_when_recipient_is_third_party():
    call = DAOFunctionCall(
        chain_id=1,
        contract_address="0x" + "ab" * 20,
        function_selector="0xa9059cbb",
        function_name="transfer",
        abi_signature="transfer(address,uint256)",
        caller_address=ALICE,
    )
    sim = SimulationResult(
        succeeded=True,
        backend="stub",
        erc20_transfers=[ERC20Transfer(token="0x" + "ab" * 20, from_addr=ALICE, to_addr=BOB, amount=100)],
    )
    flags = _post_check(call, sim)
    assert flags == []
