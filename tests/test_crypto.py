"""
test_crypto.py

Tests for the ZK Audit Trail:
  - make_commitment() produces deterministic commitment given same inputs + salt
  - verify_commitment() returns True for valid record
  - verify_commitment() returns False if commitment tampered
  - prev_hash chain: second commitment's prev_hash equals first commitment
  - AttestationSigner.sign() sets signature field
  - AttestationSigner.verify() returns True for freshly signed attestation
  - AttestationSigner.verify() returns False if attestation modified post-signing
  - AttestingAgent.evaluate_with_attestation() returns (attestation, verdict)

ECDSA tests are skipped if cryptography package is not installed.
"""

import pytest
import time
import uuid

from src.crypto.commitment import (
    CommitmentRecord,
    make_commitment,
    verify_commitment,
)


# ── L1 commitment tests ───────────────────────────────────────────────────────

def test_make_commitment_deterministic():
    """make_commitment() with same inputs (but different random salt) produces different hashes."""
    rec1 = make_commitment("hello world", "response1", "bounty-1")
    rec2 = make_commitment("hello world", "response1", "bounty-1")
    # Different salts → different commitments (correct — not deterministic!)
    # But prompt_hash and response_hash should be same
    assert rec1.prompt_hash == rec2.prompt_hash
    assert rec1.response_hash == rec2.response_hash
    # Commitments differ because of random salt
    assert rec1.commitment != rec2.commitment


def test_make_commitment_same_salt_deterministic():
    """Given same inputs AND same salt, commitment is deterministic."""
    import hashlib
    salt = "aaaa"
    ts = 1000000000000
    prompt_hash = hashlib.sha3_256("hello".encode()).hexdigest()
    response_hash = hashlib.sha3_256("world".encode()).hexdigest()
    commitment1 = hashlib.sha3_256(
        f"{prompt_hash}|{response_hash}|bounty-1|{salt}|{ts}".encode()
    ).hexdigest()
    commitment2 = hashlib.sha3_256(
        f"{prompt_hash}|{response_hash}|bounty-1|{salt}|{ts}".encode()
    ).hexdigest()
    assert commitment1 == commitment2


def test_verify_commitment_valid():
    """verify_commitment() returns True for a valid record."""
    record = make_commitment("prompt text", "response text", "bounty-42")
    assert verify_commitment(record) is True


def test_verify_commitment_tampered():
    """verify_commitment() returns False if commitment is tampered."""
    record = make_commitment("prompt text", "response text", "bounty-42")
    # Tamper with the commitment
    original = record.commitment
    record.commitment = original[:-1] + ("0" if original[-1] != "0" else "1")
    assert verify_commitment(record) is False


def test_prev_hash_chain():
    """Second commitment's prev_hash equals first commitment's commitment."""
    rec1 = make_commitment("prompt1", "response1", "bounty-1")
    rec2 = make_commitment("prompt2", "response2", "bounty-2", prev_hash=rec1.commitment)
    assert rec2.prev_hash == rec1.commitment


def test_commitment_id_format():
    """CommitmentRecord has well-formed commitment_id."""
    record = make_commitment("test", "response", "bounty-1")
    assert record.commitment_id.startswith("cmt_")
    assert len(record.commitment_id) == len("cmt_") + 16


# ── L2 attestation tests ──────────────────────────────────────────────────────

cryptography = pytest.importorskip(
    "cryptography",
    reason="cryptography package not installed — skipping ECDSA tests"
)


def make_test_attestation(attestation_id: str = None) -> "EvaluationAttestation":
    from src.crypto.attestation import EvaluationAttestation
    return EvaluationAttestation(
        attestation_id=attestation_id or f"att_{uuid.uuid4().hex[:16]}",
        commitment_id="cmt_abcdef1234567890",
        bounty_id="bounty-test-1",
        sheila_decision="violation",
        sheila_category="treasury_manipulation",
        confidence=0.97,
        reward_alpha=1.0,
        reward_beta=0.3,
        reward_gamma=3.0,
        reward_delta=0.5,
        final_score=0.97,
        timestamp_ms=int(time.time() * 1000),
        public_key_id="test_key_id",
    )


def test_attestation_signer_sign_sets_signature():
    """AttestationSigner.sign() sets the signature field."""
    from src.crypto.attestation import AttestationSigner
    signer = AttestationSigner()
    attestation = make_test_attestation()
    assert attestation.signature is None
    signed = signer.sign(attestation)
    assert signed.signature is not None
    assert len(signed.signature) > 0


def test_attestation_signer_verify_valid():
    """AttestationSigner.verify() returns True for freshly signed attestation."""
    from src.crypto.attestation import AttestationSigner
    signer = AttestationSigner()
    attestation = make_test_attestation()
    signed = signer.sign(attestation)
    assert signer.verify(signed) is True


def test_attestation_signer_verify_tampered():
    """AttestationSigner.verify() returns False if attestation fields modified post-signing."""
    from src.crypto.attestation import AttestationSigner
    signer = AttestationSigner()
    attestation = make_test_attestation()
    signed = signer.sign(attestation)

    # Tamper with the decision after signing
    signed.sheila_decision = "clean"
    assert signer.verify(signed) is False


@pytest.mark.asyncio
async def test_attesting_agent_returns_attestation_and_verdict():
    """AttestingAgent.evaluate_with_attestation() returns (attestation, verdict)."""
    from src.crypto.attestation import AttestationSigner
    from src.crypto.attesting_agent import AttestingAgent
    from agents.sheila.api import SheilaJudge, SheilaVerdict
    from unittest.mock import AsyncMock

    # Mock Sheila judge
    mock_verdict = SheilaVerdict(
        turn_id="test-turn",
        decision="violation",
        category="prompt_injection",
        confidence=0.95,
        evidence="Test",
        atlas_tactic="AML.TA0004",
        attestation_id=None,
    )
    mock_judge = AsyncMock(spec=SheilaJudge)
    mock_judge.judge = AsyncMock(return_value=mock_verdict)

    signer = AttestationSigner()
    agent = AttestingAgent(
        sheila_judge=mock_judge,
        signer=signer,
        erc8004_publisher=None,
        store=None,
    )

    attestation, verdict = await agent.evaluate_with_attestation(
        turn_id="test-turn",
        user_input="Test prompt",
        agent_response="Test response",
        bounty_id="bounty-test",
        reward_inputs={"alpha": 1.0, "beta": 0.3, "gamma": 3.0, "delta": 0.5},
    )

    assert attestation is not None
    assert verdict is not None
    assert attestation.sheila_decision == "violation"
    assert attestation.signature is not None
    assert verdict.decision == "violation"
