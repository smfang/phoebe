"""
L1: SHA3-256 Commitment / Reveal Protocol

Upgrade from Fix 2 (single-field hash) to full two-phase commit/reveal:
  - Commit: SHA3-256(prompt_hash | response_hash | bounty_id | salt | timestamp)
  - Reveal: publish pre-image; verifier recomputes hash
  - Salt: 32 random bytes per commitment (prevents rainbow table attacks)
  - bounty_id: binds commitment to specific Arena bounty

CAT-01: every evaluation produces a commitment before result is published
CAT-02: prev_hash chain links commitments in sequence (append-only log)
CAT-05: raw prompts are never stored — only their SHA3-256 hash (GDPR)
"""

import hashlib, os, time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CommitmentRecord:
    commitment_id:  str
    commitment:     str           # hex SHA3-256
    prev_hash:      Optional[str] # hash of previous commitment (CAT-02 chain)
    bounty_id:      str
    timestamp_ms:   int
    salt:           str           # hex, 32 bytes
    # reveal fields (populated on reveal phase)
    prompt_hash:    Optional[str] = None
    response_hash:  Optional[str] = None
    revealed_at_ms: Optional[int] = None


def make_commitment(
    prompt: str,
    response: str,
    bounty_id: str,
    prev_hash: Optional[str] = None,
) -> CommitmentRecord:
    """
    Phase 1: commit.
    Hash the inputs; store commitment. Raw inputs are not retained.
    """
    salt = os.urandom(32).hex()
    timestamp_ms = int(time.time() * 1000)
    prompt_hash   = hashlib.sha3_256(prompt.encode()).hexdigest()
    response_hash = hashlib.sha3_256(response.encode()).hexdigest()
    commitment    = hashlib.sha3_256(
        f"{prompt_hash}|{response_hash}|{bounty_id}|{salt}|{timestamp_ms}".encode()
    ).hexdigest()
    return CommitmentRecord(
        commitment_id=f"cmt_{commitment[:16]}",
        commitment=commitment,
        prev_hash=prev_hash,
        bounty_id=bounty_id,
        timestamp_ms=timestamp_ms,
        salt=salt,
        prompt_hash=prompt_hash,
        response_hash=response_hash,
    )


def verify_commitment(record: CommitmentRecord) -> bool:
    """Phase 2: reveal. Recompute and verify."""
    expected = hashlib.sha3_256(
        f"{record.prompt_hash}|{record.response_hash}|"
        f"{record.bounty_id}|{record.salt}|{record.timestamp_ms}".encode()
    ).hexdigest()
    return expected == record.commitment
