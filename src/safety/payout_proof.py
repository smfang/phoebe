"""
Two-phase x402 payout proof generation.

After Phoebe evaluates a submission and computes a payout, this module
generates a signed payout authorization that serves as cryptographic
proof of a verified vulnerability. The red teamer can present this
proof to claim USDC from the escrow contract or via the x402 facilitator.

Flow:
  1. Scorer evaluates submission → EvaluationResult with payout_usdc
  2. generate_payout_proof() signs (bountyId, submissionId, evaluationId,
     recipient, amount) with the arena wallet
  3. The SignedPayoutProof is returned to the red teamer
  4. Red teamer calls PhoebeEscrow.claim() with the proof, or
     presents it to the x402 facilitator for settlement

The proof is an EIP-191 personal_sign that the escrow contract or
facilitator can verify against the authorized Phoebe signer address.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SignedPayoutProof:
    """A signed payout authorization for a verified vulnerability."""

    bounty_id: str
    submission_id: str
    evaluation_id: str
    recipient: str
    amount_usdc: float
    amount_wei: int  # USDC has 6 decimals
    signature: str
    signer: str
    timestamp: int
    chain: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounty_id": self.bounty_id,
            "submission_id": self.submission_id,
            "evaluation_id": self.evaluation_id,
            "recipient": self.recipient,
            "amount_usdc": self.amount_usdc,
            "amount_wei": self.amount_wei,
            "signature": self.signature,
            "signer": self.signer,
            "timestamp": self.timestamp,
            "chain": self.chain,
        }

    def to_claim_args(self) -> dict[str, Any]:
        """Format for calling PhoebeEscrow.claim() on-chain."""
        return {
            "bountyId": f"0x{hashlib.sha256(self.bounty_id.encode()).hexdigest()}",
            "submissionId": f"0x{hashlib.sha256(self.submission_id.encode()).hexdigest()}",
            "evaluationId": self.evaluation_id,
            "amount": self.amount_wei,
            "signature": f"0x{self.signature}",
        }


def generate_payout_proof(
    wallet: Any,
    bounty_id: str,
    submission_id: str,
    evaluation_id: str,
    recipient: str,
    amount_usdc: float,
    chain: str = "base",
) -> SignedPayoutProof:
    """
    Generate a signed payout authorization.

    Uses the arena wallet (Wallet or DevWallet from src/x402/wallet.py)
    to sign the payout details. The signature can be verified on-chain
    by the PhoebeEscrow contract.

    Args:
        wallet: Wallet or DevWallet instance for signing.
        bounty_id: The bounty this payout is for.
        submission_id: The submission that earned the payout.
        evaluation_id: Oracle evaluation ID (on-chain reference).
        recipient: Red teamer's wallet address.
        amount_usdc: Payout amount in USDC.
        chain: Blockchain for the payout.

    Returns:
        SignedPayoutProof with the EIP-191 signature.
    """
    ts = int(time.time())
    amount_wei = int(amount_usdc * 1_000_000)  # USDC has 6 decimals

    # Hash the bounty/submission IDs to bytes32
    bounty_hash = hashlib.sha256(bounty_id.encode()).hexdigest()
    submission_hash = hashlib.sha256(submission_id.encode()).hexdigest()

    # Canonical message matching PhoebeEscrow.claim() verification:
    # keccak256(abi.encodePacked(bountyId, submissionId, evaluationId, recipient, amount))
    # We sign this with EIP-191 personal_sign so ecrecover works on-chain.
    message = (
        f"Phoebe Payout Authorization\n"
        f"Bounty: 0x{bounty_hash}\n"
        f"Submission: 0x{submission_hash}\n"
        f"Evaluation: {evaluation_id}\n"
        f"Recipient: {recipient}\n"
        f"Amount: {amount_wei}\n"
        f"Chain: {chain}\n"
        f"Timestamp: {ts}"
    )

    # Use the wallet's signing capability
    from src.x402.wallet import DevWallet, Wallet

    if isinstance(wallet, Wallet):
        from eth_account.messages import encode_defunct

        signable = encode_defunct(text=message)
        signed = wallet._account.sign_message(signable)
        signature = signed.signature.hex()
        signer = wallet.address
    elif isinstance(wallet, DevWallet):
        import hmac as _hmac

        sig = _hmac.new(
            b"dev-key", message.encode(), hashlib.sha256
        ).hexdigest()
        signature = sig
        signer = wallet.address
    else:
        raise TypeError(f"Unsupported wallet type: {type(wallet)}")

    proof = SignedPayoutProof(
        bounty_id=bounty_id,
        submission_id=submission_id,
        evaluation_id=evaluation_id,
        recipient=recipient,
        amount_usdc=amount_usdc,
        amount_wei=amount_wei,
        signature=signature,
        signer=signer,
        timestamp=ts,
        chain=chain,
    )

    logger.info(
        "Payout proof generated: %s USDC to %s for submission %s",
        amount_usdc, recipient[:16], submission_id,
    )

    return proof


def generate_evaluation_receipt(
    wallet: Any,
    submission_id: str,
    prompt_hash: str,
    category: str,
    severity: int,
    unsafe: bool,
    score: float,
) -> dict[str, Any]:
    """
    Generate a signed evaluation receipt (lightweight, no payout).

    This is the proof that Phoebe evaluated a specific prompt and
    classified it. Used as a stage-gate credential: the receipt
    proves the attack was evaluated, allowing progression to the
    next pipeline stage.

    Returns a dict with the receipt data and signature.
    """
    ts = int(time.time())

    receipt_data = {
        "submission_id": submission_id,
        "prompt_hash": prompt_hash,
        "category": category,
        "severity": severity,
        "unsafe": unsafe,
        "score": round(score, 6),
        "timestamp": ts,
    }

    canonical = json.dumps(receipt_data, sort_keys=True, separators=(",", ":"))

    from src.x402.wallet import DevWallet, Wallet

    if isinstance(wallet, Wallet):
        from eth_account.messages import encode_defunct

        signable = encode_defunct(text=canonical)
        signed = wallet._account.sign_message(signable)
        signature = signed.signature.hex()
        signer = wallet.address
    elif isinstance(wallet, DevWallet):
        import hmac as _hmac

        signature = _hmac.new(
            b"dev-key", canonical.encode(), hashlib.sha256
        ).hexdigest()
        signer = wallet.address
    else:
        raise TypeError(f"Unsupported wallet type: {type(wallet)}")

    return {
        **receipt_data,
        "signature": signature,
        "signer": signer,
    }
