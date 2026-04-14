"""Typed models for the DAO domain module."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DAOFunctionCall(BaseModel):
    """A single on-chain function call submitted by an agent for verification."""

    chain_id: int
    contract_address: str
    function_selector: str  # e.g. "0xa9059cbb"
    function_name: str  # e.g. "transfer"
    abi_signature: str  # e.g. "transfer(address,uint256)"
    decoded_params: dict[str, Any] = Field(default_factory=dict)
    raw_calldata: str = ""
    value_wei: int = 0
    gas_limit: int | None = None
    caller_address: str = ""
    user_intent: str = ""

    def normalized_address(self) -> str:
        return self.contract_address.lower()


class ValidationResult(BaseModel):
    """Output of `FunctionCallValidator`."""

    valid: bool
    matched_schema: str | None = None
    violations: list[str] = Field(default_factory=list)


class ERC20Transfer(BaseModel):
    token: str
    from_addr: str
    to_addr: str
    amount: int


class OwnershipChange(BaseModel):
    contract: str
    previous_owner: str
    new_owner: str


class ProxyUpgrade(BaseModel):
    proxy: str
    previous_implementation: str
    new_implementation: str


class SimulationResult(BaseModel):
    """Output of `SimulationRunner`."""

    succeeded: bool
    revert_reason: str | None = None
    gas_used: int = 0
    backend: str = ""
    state_diff: dict[str, dict[str, str]] = Field(default_factory=dict)
    balance_changes: dict[str, int] = Field(default_factory=dict)
    erc20_transfers: list[ERC20Transfer] = Field(default_factory=list)
    ownership_changes: list[OwnershipChange] = Field(default_factory=list)
    proxy_upgrades: list[ProxyUpgrade] = Field(default_factory=list)
    suspicious_flags: list[str] = Field(default_factory=list)


class AlignmentResult(BaseModel):
    """Output of `SemanticAlignmentJudge`."""

    aligned: bool
    confidence: float = 0.0
    severity: int = 0
    intent_summary: str = ""
    actual_summary: str = ""
    explanation: str = ""
