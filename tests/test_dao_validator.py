"""Tests for the DAO FunctionCallValidator stage."""

import pytest

from src.domains.base import Action, EvalRequest, EvalState, RiskTier
from src.domains.dao.validator import FunctionCallValidator, MAX_UINT256, ZERO_ADDRESS

ALICE = "0x" + "11" * 20
BOB = "0x" + "22" * 20
ATTACKER = "0x" + "ff" * 20


def _state(tier: RiskTier = RiskTier.T2) -> EvalState:
    return EvalState(
        request_id="r1", domain="dao", module_version="0.1.0", tier=tier
    )


def _req(payload: dict) -> EvalRequest:
    return EvalRequest(domain="dao", payload=payload)


def _good_transfer(amount: int = 100) -> dict:
    return {
        "chain_id": 1,
        "contract_address": "0x" + "ab" * 20,
        "function_selector": "0xa9059cbb",
        "function_name": "transfer",
        "abi_signature": "transfer(address,uint256)",
        "decoded_params": {"to": BOB, "amount": amount},
        "raw_calldata": "0xa9059cbb" + "00" * 64,
        "caller_address": ALICE,
    }


@pytest.mark.asyncio
async def test_valid_transfer_passes():
    v = FunctionCallValidator()
    result = await v.run(_req(_good_transfer()), _state())
    assert result.action == Action.PASS
    assert result.severity == 0


@pytest.mark.asyncio
async def test_unparseable_payload_blocks():
    v = FunctionCallValidator()
    result = await v.run(_req({"junk": True}), _state())
    assert result.action == Action.BLOCK
    assert result.severity == 4


@pytest.mark.asyncio
async def test_invalid_contract_address():
    v = FunctionCallValidator()
    payload = _good_transfer()
    payload["contract_address"] = "not-an-address"
    result = await v.run(_req(payload), _state())
    assert result.action == Action.BLOCK
    assert "contract_address" in result.reason


@pytest.mark.asyncio
async def test_unknown_signature_blocks_with_high_severity():
    v = FunctionCallValidator()
    payload = _good_transfer()
    payload["abi_signature"] = "rugPull(uint256)"
    result = await v.run(_req(payload), _state())
    assert result.action == Action.BLOCK
    assert result.severity == 4


@pytest.mark.asyncio
async def test_zero_address_transfer_blocks():
    v = FunctionCallValidator()
    payload = _good_transfer()
    payload["decoded_params"]["to"] = ZERO_ADDRESS
    result = await v.run(_req(payload), _state())
    assert result.action == Action.BLOCK
    assert "zero address" in result.reason


@pytest.mark.asyncio
async def test_calldata_selector_mismatch():
    v = FunctionCallValidator()
    payload = _good_transfer()
    payload["raw_calldata"] = "0xdeadbeef" + "00" * 64
    result = await v.run(_req(payload), _state())
    assert result.action == Action.BLOCK
    assert "selector" in result.reason


@pytest.mark.asyncio
async def test_scammer_recipient_blocks():
    v = FunctionCallValidator(scammer_addresses={ATTACKER})
    payload = _good_transfer()
    payload["decoded_params"]["to"] = ATTACKER
    result = await v.run(_req(payload), _state())
    assert result.action == Action.BLOCK
    assert "scammer" in result.reason


@pytest.mark.asyncio
async def test_scammer_caller_blocks():
    v = FunctionCallValidator(scammer_addresses={ATTACKER})
    payload = _good_transfer()
    payload["caller_address"] = ATTACKER
    result = await v.run(_req(payload), _state())
    assert result.action == Action.BLOCK


@pytest.mark.asyncio
async def test_infinite_approval_flagged():
    v = FunctionCallValidator()
    payload = _good_transfer()
    payload["function_selector"] = "0x095ea7b3"
    payload["function_name"] = "approve"
    payload["abi_signature"] = "approve(address,uint256)"
    payload["decoded_params"] = {"spender": BOB, "amount": MAX_UINT256}
    payload["raw_calldata"] = "0x095ea7b3" + "00" * 64
    result = await v.run(_req(payload), _state())
    assert result.action == Action.BLOCK
    assert "infinite" in result.reason.lower()


@pytest.mark.asyncio
async def test_state_scratch_populated_for_downstream():
    v = FunctionCallValidator()
    state = _state()
    await v.run(_req(_good_transfer()), state)
    assert "call" in state.scratch
    assert state.scratch["call"]["function_name"] == "transfer"


@pytest.mark.asyncio
async def test_invalid_calldata_hex():
    v = FunctionCallValidator()
    payload = _good_transfer()
    payload["raw_calldata"] = "not-hex"
    result = await v.run(_req(payload), _state())
    assert result.action == Action.BLOCK
    assert "hex" in result.reason
