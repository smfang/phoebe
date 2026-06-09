"""
Sara Cryptographic Audit Trail — three layers:

L1  SHA3-256 commit/reveal with salt + bounty_id   (this file)
L2  ECDSA P-384 EvaluationAttestation               (attestation.py)
L3  SP1 STARK proofs                                (zk_proof.py — Phase 5)

All evaluations produce at minimum an L1 commitment.
L2 attestation is produced for all Sheila judge verdicts.
L3 ZK proof is optional, gated on ZK_PROOF_ENABLED env var.
"""
