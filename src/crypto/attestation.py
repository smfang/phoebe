"""
L2: EvaluationAttestation — ECDSA P-384 signed evaluation record.

Every Sheila judge verdict produces a signed EvaluationAttestation.
The attestation binds:
  - the commitment_id (L1)
  - Sheila's verdict (decision, category, confidence)
  - the scoring formula inputs (α/β/γ/δ)
  - timestamp

Signature key: ECDSA P-384 (NIST curve, FIPS 186-4 compliant)
Post-quantum path: ML-DSA (CRYSTALS-Dilithium) alongside P-384 — Phase 5

Verifiers (smart contracts, auditors, Regen Bio) can verify attestations
without re-running the evaluation — they only need the public key.
"""

import json, time, hashlib
from dataclasses import dataclass, asdict
from typing import Optional

try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


@dataclass
class EvaluationAttestation:
    attestation_id:   str
    commitment_id:    str           # links to L1 CommitmentRecord
    bounty_id:        str
    sheila_decision:  str           # violation | clean | borderline
    sheila_category:  Optional[str]
    confidence:       float
    reward_alpha:     float         # α component of RewardComputer
    reward_beta:      float         # β component
    reward_gamma:     float         # γ component (FN penalty)
    reward_delta:     float         # δ component (FP penalty)
    final_score:      float
    timestamp_ms:     int
    public_key_id:    str           # fingerprint of signing key
    signature:        Optional[str] = None   # hex DER; None until signed


class AttestationSigner:
    """
    Signs EvaluationAttestations with ECDSA P-384.
    Key is loaded from SHEILA_ATTESTATION_KEY_PEM env var or generated fresh.

    In Phase 5 TEE: private key never leaves the enclave.
    For now: key is stored in env var / secrets manager.
    """

    def __init__(self):
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography package required: pip install cryptography"
            )
        import os
        pem = os.getenv("SHEILA_ATTESTATION_KEY_PEM")
        if pem:
            self._private_key = serialization.load_pem_private_key(
                pem.encode(), password=None, backend=default_backend()
            )
        else:
            # Generate ephemeral key (dev/test only — log a warning)
            import logging
            logging.getLogger("sara.crypto").warning(
                "SHEILA_ATTESTATION_KEY_PEM not set — using ephemeral key. "
                "Set this env var in production."
            )
            self._private_key = ec.generate_private_key(
                ec.SECP384R1(), default_backend()
            )
        self._public_key = self._private_key.public_key()
        self.key_id = self._compute_key_id()

    def sign(self, attestation: EvaluationAttestation) -> EvaluationAttestation:
        """Sign the attestation. Returns a new attestation with signature set."""
        payload = self._canonical_payload(attestation)
        sig = self._private_key.sign(
            payload.encode(),
            ec.ECDSA(hashes.SHA384())
        )
        attestation.signature = sig.hex()
        return attestation

    def verify(self, attestation: EvaluationAttestation) -> bool:
        """Verify an attestation signature."""
        if not attestation.signature:
            return False
        payload = self._canonical_payload(attestation)
        try:
            self._public_key.verify(
                bytes.fromhex(attestation.signature),
                payload.encode(),
                ec.ECDSA(hashes.SHA384())
            )
            return True
        except Exception:
            return False

    def _canonical_payload(self, a: EvaluationAttestation) -> str:
        """Deterministic string representation for signing."""
        return (
            f"{a.attestation_id}|{a.commitment_id}|{a.bounty_id}|"
            f"{a.sheila_decision}|{a.sheila_category}|{a.confidence:.6f}|"
            f"{a.final_score:.6f}|{a.timestamp_ms}"
        )

    def _compute_key_id(self) -> str:
        pub_bytes = self._public_key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return hashlib.sha256(pub_bytes).hexdigest()[:16]

    def public_key_pem(self) -> str:
        return self._public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
