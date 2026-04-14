"""Tests for the DAO SemanticAlignmentJudge stage."""

import pytest

from src.domains.base import Action, EvalRequest, EvalState, RiskTier
from src.domains.dao.alignment import SemanticAlignmentJudge, _build_prompt, _parse_response
from src.domains.dao.models import DAOFunctionCall, SimulationResult


ALICE = "0x" + "11" * 20
BOB = "0x" + "22" * 20

CALL_PAYLOAD = {
    "chain_id": 1,
    "contract_address": "0x" + "ab" * 20,
    "function_selector": "0xa9059cbb",
    "function_name": "transfer",
    "abi_signature": "transfer(address,uint256)",
    "decoded_params": {"to": BOB, "amount": 100},
    "raw_calldata": "0xa9059cbb",
    "caller_address": ALICE,
}


def _state(tier: RiskTier = RiskTier.T2) -> EvalState:
    s = EvalState(request_id="r1", domain="dao", module_version="0.1.0", tier=tier)
    s.scratch["call"] = CALL_PAYLOAD
    return s


def _req(intent: str = "send 100 USDC to bob") -> EvalRequest:
    return EvalRequest(domain="dao", payload=CALL_PAYLOAD, user_intent=intent)


class _StubClassifier:
    def __init__(self, response: dict | None = None, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.calls: list[dict] = []

    async def classify(self, prompt, model_output, category):
        self.calls.append({"prompt": prompt, "model_output": model_output, "category": category})
        if self._raises:
            raise self._raises
        return self._response


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t1_skipped():
    judge = SemanticAlignmentJudge(classifier=_StubClassifier())
    result = await judge.run(_req(), _state(tier=RiskTier.T1))
    assert result.action == Action.PASS
    assert "skipped" in result.reason


@pytest.mark.asyncio
async def test_aligned_response_passes():
    classifier = _StubClassifier(response={
        "aligned": True,
        "confidence": 0.95,
        "intent_summary": "send 100 USDC to bob",
        "actual_summary": "transfers 100 to bob",
        "explanation": "match",
    })
    judge = SemanticAlignmentJudge(classifier=classifier)
    result = await judge.run(_req(), _state())
    assert result.action == Action.PASS
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_misaligned_response_blocks():
    classifier = _StubClassifier(response={
        "aligned": False,
        "confidence": 0.9,
        "severity": 4,
        "intent_summary": "send 100 USDC to bob",
        "actual_summary": "transfers to attacker wallet",
        "explanation": "recipient mismatch",
    })
    judge = SemanticAlignmentJudge(classifier=classifier)
    result = await judge.run(_req(), _state())
    assert result.action == Action.BLOCK
    assert result.severity == 4
    assert "mismatch" in result.reason


@pytest.mark.asyncio
async def test_classifier_unsafe_shape_inverted_to_misaligned():
    # SafetyClassifier's day-1 schema uses `unsafe`, not `aligned`.
    classifier = _StubClassifier(response={
        "unsafe": True,
        "severity": 5,
        "explanation": "looks like memory injection",
    })
    judge = SemanticAlignmentJudge(classifier=classifier)
    result = await judge.run(_req(), _state())
    assert result.action == Action.BLOCK
    assert result.severity == 5


@pytest.mark.asyncio
async def test_classifier_failure_is_soft_pass():
    classifier = _StubClassifier(raises=RuntimeError("api down"))
    judge = SemanticAlignmentJudge(classifier=classifier)
    result = await judge.run(_req(), _state())
    assert result.action == Action.PASS
    assert "judge unreachable" in result.reason


@pytest.mark.asyncio
async def test_judge_receives_dao_category():
    classifier = _StubClassifier(response={"aligned": True})
    judge = SemanticAlignmentJudge(classifier=classifier, category="dao_function_call")
    await judge.run(_req(), _state())
    assert classifier.calls[0]["category"] == "dao_function_call"


def test_build_prompt_includes_intent_and_call():
    call = DAOFunctionCall(**CALL_PAYLOAD)
    prompt = _build_prompt("transfer 100 USDC to bob", call, sim=None)
    assert "transfer 100 USDC to bob" in prompt
    assert "transfer(address,uint256)" in prompt
    assert "## Decoded Function Call" in prompt


def test_build_prompt_includes_simulation_summary_when_present():
    call = DAOFunctionCall(**CALL_PAYLOAD)
    sim = SimulationResult(succeeded=True, gas_used=21000, backend="stub")
    prompt = _build_prompt("send", call, sim=sim)
    assert "## Simulation Summary" in prompt
    assert "21000" in prompt


def test_parse_response_native_shape():
    result = _parse_response({
        "aligned": False,
        "confidence": 0.8,
        "severity": 3,
        "explanation": "off",
    })
    assert result.aligned is False
    assert result.severity == 3


def test_parse_response_unsafe_shape():
    result = _parse_response({"unsafe": True, "severity": 4, "explanation": "hmm"})
    assert result.aligned is False
    assert result.severity == 4
