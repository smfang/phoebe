"""Stage 1: deterministic schema + parameter validation for DAO calls.

This stage catches the cheapest class of malicious calls — wrong selector,
calldata that doesn't match decoded params, infinite approvals to EOAs,
zero-address transfers — without paying for simulation or LLM judgment.
"""

from __future__ import annotations

import re
import time
from typing import Any

from src.domains.base import Action, EvalRequest, EvalState, StageResult
from src.domains.dao.models import DAOFunctionCall, ValidationResult
from src.domains.dao.risk_tiers import FUNCTION_TIERS

ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
SELECTOR_RE = re.compile(r"^0x[a-fA-F0-9]{8}$")
HEX_RE = re.compile(r"^0x[a-fA-F0-9]*$")
ZERO_ADDRESS = "0x" + "0" * 40
MAX_UINT256 = 2**256 - 1


class FunctionCallValidator:
    """Verifies a DAO function call against schema + parameter rules."""

    name = "validator"

    def __init__(
        self,
        known_signatures: dict[str, Any] | None = None,
        scammer_addresses: set[str] | None = None,
    ) -> None:
        # Default to the FUNCTION_TIERS keys as the allow-list of recognized signatures.
        self._signatures = known_signatures or {sig: True for sig in FUNCTION_TIERS}
        self._scammers = {a.lower() for a in (scammer_addresses or set())}

    async def run(self, req: EvalRequest, state: EvalState) -> StageResult:
        started = time.perf_counter()
        violations: list[str] = []

        try:
            call = DAOFunctionCall(**req.payload)
        except Exception as exc:
            return StageResult(
                stage=self.name,
                action=Action.BLOCK,
                severity=4,
                reason="payload is not a valid DAOFunctionCall",
                details={"error": str(exc)},
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        # Persist the parsed call for downstream stages
        state.scratch["call"] = call.model_dump()

        # ── address sanity ────────────────────────────────────────────────
        if not ADDRESS_RE.match(call.contract_address):
            violations.append("contract_address is not a valid 0x address")
        if call.caller_address and not ADDRESS_RE.match(call.caller_address):
            violations.append("caller_address is not a valid 0x address")

        # ── selector + abi consistency ────────────────────────────────────
        if not SELECTOR_RE.match(call.function_selector):
            violations.append("function_selector must be 4 bytes (0x + 8 hex chars)")
        if call.raw_calldata and not HEX_RE.match(call.raw_calldata):
            violations.append("raw_calldata is not valid hex")
        if call.raw_calldata and call.function_selector and not call.raw_calldata.lower().startswith(
            call.function_selector.lower()
        ):
            violations.append("raw_calldata does not begin with function_selector")

        # ── known signature ───────────────────────────────────────────────
        matched = call.abi_signature if call.abi_signature in self._signatures else None
        if matched is None:
            violations.append(f"unknown function signature: {call.abi_signature}")

        # ── caller authorization on the scammer list ──────────────────────
        if call.caller_address.lower() in self._scammers:
            violations.append("caller is on the known-scammer list")

        # ── per-function parameter rules ──────────────────────────────────
        if matched is not None:
            violations.extend(self._check_params(call))

        result = ValidationResult(
            valid=not violations,
            matched_schema=matched,
            violations=violations,
        )

        action = Action.PASS if result.valid else Action.BLOCK
        # Severity scales with violation type; unknown signatures and zero-address
        # transfers earn a higher severity than mere format errors.
        severity = 0 if result.valid else (4 if any("unknown" in v or "zero" in v for v in violations) else 2)

        return StageResult(
            stage=self.name,
            action=action,
            severity=severity,
            reason="; ".join(violations) if violations else "schema + parameters valid",
            details=result.model_dump(),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ------------------------------------------------------------------
    # Per-function parameter checks
    # ------------------------------------------------------------------

    def _check_params(self, call: DAOFunctionCall) -> list[str]:
        sig = call.abi_signature
        params = call.decoded_params
        violations: list[str] = []

        if sig in {"transfer(address,uint256)", "transferFrom(address,address,uint256)"}:
            recipient = str(params.get("to", "")).lower()
            if recipient == ZERO_ADDRESS:
                violations.append("transfer to the zero address")
            if recipient and recipient in self._scammers:
                violations.append("transfer recipient is on the known-scammer list")
            amount = int(params.get("amount", 0) or 0)
            if amount < 0:
                violations.append("transfer amount is negative")
            if amount > MAX_UINT256:
                violations.append("transfer amount exceeds uint256")

        elif sig == "approve(address,uint256)":
            spender = str(params.get("spender", "")).lower()
            amount = int(params.get("amount", 0) or 0)
            if spender == ZERO_ADDRESS:
                violations.append("approval to the zero address")
            # Infinite approval flags as a violation; tier should already be T3,
            # but this gives the validator a chance to surface it explicitly.
            if amount == MAX_UINT256:
                violations.append("infinite approval (amount = MaxUint256)")

        elif sig in {"upgradeTo(address)", "upgradeToAndCall(address,bytes)", "changeAdmin(address)"}:
            target = str(params.get("newImplementation", params.get("newAdmin", ""))).lower()
            if target == ZERO_ADDRESS:
                violations.append("admin/implementation change to the zero address")

        elif sig in {"transferOwnership(address)"}:
            new_owner = str(params.get("newOwner", "")).lower()
            if new_owner == ZERO_ADDRESS:
                violations.append("ownership transfer to the zero address")

        return violations
