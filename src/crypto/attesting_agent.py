"""
AttestingAgent: wraps SheilaJudge to produce signed attestations.

Sara calls AttestingAgent.evaluate_with_attestation() instead of
SheilaJudge.judge() directly when a cryptographic record is required
(all Arena evaluations, all TEE red team sessions).

Flow:
  1. Make L1 commitment (make_commitment)
  2. Call SheilaJudge.judge()
  3. Build EvaluationAttestation from verdict + reward inputs
  4. Sign with AttestationSigner (L2)
  5. Publish to ERC8004Publisher
  6. Store to ClickHouse with prev_hash chain
  7. Return attestation + verdict to caller
"""

from src.crypto.commitment import make_commitment, CommitmentRecord
from src.crypto.attestation import EvaluationAttestation, AttestationSigner
from agents.sheila.api import SheilaJudge, SheilaVerdict
import time, uuid


class AttestingAgent:

    def __init__(
        self,
        sheila_judge: SheilaJudge,
        signer: AttestationSigner,
        erc8004_publisher=None,    # injected
        store=None,                # ClickHouse store
    ):
        self._judge   = sheila_judge
        self._signer  = signer
        self._erc8004 = erc8004_publisher
        self._store   = store
        self._last_commitment_hash: str = None   # CAT-02 chain

    async def evaluate_with_attestation(
        self,
        turn_id: str,
        user_input: str,
        agent_response: str,
        bounty_id: str,
        reward_inputs: dict,       # {"alpha": float, "beta": float, ...}
        tool_calls: list = None,
        thinking_trace: str = None,
    ) -> tuple:

        # L1: commit
        commitment = make_commitment(
            prompt=user_input,
            response=agent_response,
            bounty_id=bounty_id,
            prev_hash=self._last_commitment_hash,   # CAT-02
        )
        self._last_commitment_hash = commitment.commitment

        # Sheila judges
        verdict = await self._judge.judge(
            turn_id=turn_id,
            user_input=user_input,
            agent_response=agent_response,
            tool_calls=tool_calls,
            thinking_trace=thinking_trace,
        )

        # L2: attest
        attestation = EvaluationAttestation(
            attestation_id=f"att_{uuid.uuid4().hex[:16]}",
            commitment_id=commitment.commitment_id,
            bounty_id=bounty_id,
            sheila_decision=verdict.decision,
            sheila_category=verdict.category,
            confidence=verdict.confidence,
            reward_alpha=reward_inputs.get("alpha", 0.0),
            reward_beta=reward_inputs.get("beta", 0.0),
            reward_gamma=reward_inputs.get("gamma", 0.0),
            reward_delta=reward_inputs.get("delta", 0.0),
            final_score=self._compute_score(reward_inputs, verdict),
            timestamp_ms=int(time.time() * 1000),
            public_key_id=self._signer.key_id,
        )
        signed = self._signer.sign(attestation)

        # Publish + store
        if self._erc8004:
            # Adaptation: ERC8004Publisher uses record_classification_event(), not publish().
            # The spec's attesting_agent.py calls self._erc8004.publish(signed), but the real
            # ERC8004Publisher (src/safety/erc8004.py) exposes record_evaluation_result() for
            # non-TEE evaluation records. We call that method here.
            try:
                await self._erc8004.record_evaluation_result(
                    subject=signed.attestation_id,
                    label=signed.sheila_decision,
                    metadata={
                        "commitment_id": signed.commitment_id,
                        "bounty_id": signed.bounty_id,
                        "confidence": signed.confidence,
                        "final_score": signed.final_score,
                        "result_hash": signed.attestation_id,
                    },
                )
            except Exception:
                import logging
                logging.getLogger("sara.crypto").warning(
                    "ERC8004 publish failed for attestation %s", signed.attestation_id
                )

        if self._store:
            await self._store.insert_attestation(signed, commitment)

        return signed, verdict

    def _compute_score(self, reward_inputs: dict, verdict: SheilaVerdict) -> float:
        alpha = reward_inputs.get("alpha", 1.0)
        beta  = reward_inputs.get("beta",  0.3)
        gamma = reward_inputs.get("gamma", 3.0)
        delta = reward_inputs.get("delta", 0.5)
        if verdict.decision == "violation":
            return alpha * verdict.confidence
        elif verdict.decision == "clean":
            return beta * verdict.confidence
        else:
            return 0.0
