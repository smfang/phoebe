"""Risk-tier classification for DAO function calls.

Tier assignment combines a static lookup (function signature → default tier)
with parametric upgrades (e.g. transfer < threshold = T2, transfer >= threshold
= T3). Anything not in the registry defaults to T4 by policy — unknown call =
highest risk.
"""

from __future__ import annotations

from typing import Callable

from src.domains.base import EnforcementMode, EvalRequest, RiskTier
from src.domains.dao.models import DAOFunctionCall

# Threshold below which token transfers are treated as routine (T2 vs T3).
# Stored in the smallest unit (USDC has 6 decimals → 1000 USDC = 1_000_000_000).
DEFAULT_T2_AMOUNT_THRESHOLD = 1_000 * 10**6


# A tier function takes a parsed DAOFunctionCall and returns a RiskTier.
TierFn = Callable[[DAOFunctionCall], RiskTier]


def _transfer_tier(threshold: int) -> TierFn:
    def _fn(call: DAOFunctionCall) -> RiskTier:
        amount = int(call.decoded_params.get("amount", 0) or 0)
        return RiskTier.T2 if amount < threshold else RiskTier.T3

    return _fn


# ---------------------------------------------------------------------------
# Default registry — function ABI signature → tier (or parametric function)
# ---------------------------------------------------------------------------

FUNCTION_TIERS: dict[str, RiskTier | TierFn] = {
    # ── T1: read-only ──────────────────────────────────────────────────────
    "balanceOf(address)": RiskTier.T1,
    "totalSupply()": RiskTier.T1,
    "getVotes(address)": RiskTier.T1,
    "name()": RiskTier.T1,
    "symbol()": RiskTier.T1,
    "decimals()": RiskTier.T1,
    "owner()": RiskTier.T1,
    # ── T2/T3: value-bearing (parametric) ──────────────────────────────────
    "transfer(address,uint256)": _transfer_tier(DEFAULT_T2_AMOUNT_THRESHOLD),
    "transferFrom(address,address,uint256)": _transfer_tier(DEFAULT_T2_AMOUNT_THRESHOLD),
    # ── T3: governance + approvals ─────────────────────────────────────────
    "approve(address,uint256)": RiskTier.T3,
    "castVote(uint256,uint8)": RiskTier.T3,
    "castVoteWithReason(uint256,uint8,string)": RiskTier.T3,
    "delegate(address)": RiskTier.T3,
    "propose(address[],uint256[],bytes[],string)": RiskTier.T3,
    # ── T4: privileged / admin / irreversible ──────────────────────────────
    "upgradeTo(address)": RiskTier.T4,
    "upgradeToAndCall(address,bytes)": RiskTier.T4,
    "transferOwnership(address)": RiskTier.T4,
    "renounceOwnership()": RiskTier.T4,
    "grantRole(bytes32,address)": RiskTier.T4,
    "revokeRole(bytes32,address)": RiskTier.T4,
    "changeAdmin(address)": RiskTier.T4,
    "execute(address[],uint256[],bytes[])": RiskTier.T4,
}


# Tier → enforcement mode applied by the orchestrator.
DAO_TIER_MAP: dict[RiskTier, EnforcementMode] = {
    RiskTier.T1: EnforcementMode.ASYNC,
    RiskTier.T2: EnforcementMode.QUARANTINE,
    RiskTier.T3: EnforcementMode.SYNC,
    RiskTier.T4: EnforcementMode.SYNC_MULTISIG,
}


def classify_call(call: DAOFunctionCall) -> RiskTier:
    """Resolve a DAOFunctionCall to its risk tier.

    Unknown signatures default to T4 (unknown = privileged-by-policy).
    """
    entry = FUNCTION_TIERS.get(call.abi_signature)
    if entry is None:
        return RiskTier.T4
    if isinstance(entry, RiskTier):
        return entry
    # Parametric (callable) tier function
    return entry(call)


def dao_risk_tier(req: EvalRequest) -> RiskTier:
    """RiskTierFn for `DomainModule.risk_tier_fn`.

    Parses the request's payload into a `DAOFunctionCall` and classifies it.
    Malformed payloads are treated as T4 (highest risk → fail closed).
    """
    try:
        call = DAOFunctionCall(**req.payload)
    except Exception:
        return RiskTier.T4
    return classify_call(call)
