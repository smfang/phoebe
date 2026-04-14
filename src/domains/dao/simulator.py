"""Stage 2: execute a DAO call against forked chain state and observe effects.

Default backend speaks Anvil's JSON-RPC over httpx. The backend is pluggable
so that production deployments can swap in Tenderly or a hosted Foundry
forking service via the `SimulationBackend` protocol.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import httpx

from src.domains.base import Action, EnforcementMode, EvalRequest, EvalState, StageResult
from src.domains.dao.models import (
    DAOFunctionCall,
    ERC20Transfer,
    OwnershipChange,
    ProxyUpgrade,
    SimulationResult,
)
from src.domains.dao.risk_tiers import classify_call

# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class SimulationBackend(Protocol):
    name: str

    async def simulate(self, call: DAOFunctionCall) -> SimulationResult: ...


# Topic hashes for the events we care about.
ERC20_TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
OWNERSHIP_TRANSFERRED_TOPIC = "0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0"
PROXY_UPGRADED_TOPIC = "0xbc7cd75a20ee27fd9adebab32041f755214dbc6bffa90cc0225b39da2e5c2d3b"


class AnvilBackend:
    """Talks to a local or remote Anvil instance via JSON-RPC."""

    name = "anvil"

    def __init__(
        self,
        rpc_url: str = "http://localhost:8545",
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._rpc_url = rpc_url
        self._http = http or httpx.AsyncClient(timeout=30.0)

    async def simulate(self, call: DAOFunctionCall) -> SimulationResult:
        # Impersonate the caller so the call doesn't need a signature
        await self._rpc("anvil_impersonateAccount", [call.caller_address or "0x" + "0" * 40])

        tx = {
            "from": call.caller_address or "0x" + "0" * 40,
            "to": call.contract_address,
            "data": call.raw_calldata,
            "value": hex(call.value_wei),
        }
        if call.gas_limit:
            tx["gas"] = hex(call.gas_limit)

        try:
            tx_hash = await self._rpc("eth_sendTransaction", [tx])
            receipt = await self._rpc("eth_getTransactionReceipt", [tx_hash])
        except _RPCRevert as rev:
            return SimulationResult(
                succeeded=False,
                revert_reason=rev.message,
                backend=self.name,
            )
        except httpx.HTTPError as exc:
            raise _BackendUnreachable(str(exc)) from exc

        return _parse_receipt(receipt, backend=self.name)

    async def _rpc(self, method: str, params: list[Any]) -> Any:
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        resp = await self._http.post(self._rpc_url, json=body)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            err = data["error"]
            msg = err.get("message", "rpc error") if isinstance(err, dict) else str(err)
            if "revert" in msg.lower():
                raise _RPCRevert(msg)
            raise httpx.HTTPError(msg)
        return data.get("result")


class _BackendUnreachable(RuntimeError):
    pass


class _RPCRevert(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _parse_receipt(receipt: dict[str, Any] | None, backend: str) -> SimulationResult:
    """Translate an Ethereum receipt into a SimulationResult."""
    if not receipt:
        return SimulationResult(succeeded=False, revert_reason="no receipt", backend=backend)

    succeeded = str(receipt.get("status", "0x1")).lower() in {"0x1", "1", "true"}
    gas_used = int(receipt.get("gasUsed", "0x0"), 16) if isinstance(receipt.get("gasUsed"), str) else int(receipt.get("gasUsed", 0))

    erc20: list[ERC20Transfer] = []
    ownership: list[OwnershipChange] = []
    proxy: list[ProxyUpgrade] = []
    suspicious: list[str] = []
    balance_changes: dict[str, int] = {}

    for log in receipt.get("logs", []) or []:
        topics = log.get("topics", [])
        if not topics:
            continue
        topic0 = topics[0].lower()

        if topic0 == ERC20_TRANSFER_TOPIC and len(topics) >= 3:
            from_addr = "0x" + topics[1][-40:]
            to_addr = "0x" + topics[2][-40:]
            data = log.get("data", "0x0")
            amount = int(data, 16) if isinstance(data, str) and data.startswith("0x") else 0
            erc20.append(ERC20Transfer(
                token=log.get("address", ""),
                from_addr=from_addr,
                to_addr=to_addr,
                amount=amount,
            ))
            balance_changes[from_addr] = balance_changes.get(from_addr, 0) - amount
            balance_changes[to_addr] = balance_changes.get(to_addr, 0) + amount

        elif topic0 == OWNERSHIP_TRANSFERRED_TOPIC and len(topics) >= 3:
            ownership.append(OwnershipChange(
                contract=log.get("address", ""),
                previous_owner="0x" + topics[1][-40:],
                new_owner="0x" + topics[2][-40:],
            ))
            suspicious.append("ownership transferred")

        elif topic0 == PROXY_UPGRADED_TOPIC and len(topics) >= 2:
            proxy.append(ProxyUpgrade(
                proxy=log.get("address", ""),
                previous_implementation="",
                new_implementation="0x" + topics[1][-40:],
            ))
            suspicious.append("proxy implementation upgraded")

    return SimulationResult(
        succeeded=succeeded,
        gas_used=gas_used,
        backend=backend,
        balance_changes=balance_changes,
        erc20_transfers=erc20,
        ownership_changes=ownership,
        proxy_upgrades=proxy,
        suspicious_flags=suspicious,
    )


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class SimulationRunner:
    """Runs the call in a sandboxed fork and surfaces post-conditions."""

    name = "simulator"

    def __init__(
        self,
        backend: SimulationBackend,
        skip_for_t1: bool = True,
        fail_closed_tiers: tuple[str, ...] = ("T3", "T4"),
    ) -> None:
        self._backend = backend
        self._skip_for_t1 = skip_for_t1
        self._fail_closed_tiers = set(fail_closed_tiers)

    async def run(self, req: EvalRequest, state: EvalState) -> StageResult:
        started = time.perf_counter()

        if self._skip_for_t1 and state.tier.value == "T1":
            return StageResult(
                stage=self.name,
                action=Action.PASS,
                reason="skipped (T1)",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        try:
            call = DAOFunctionCall(**(state.scratch.get("call") or req.payload))
        except Exception as exc:
            return StageResult(
                stage=self.name,
                action=Action.BLOCK,
                severity=4,
                reason="payload not parseable as DAOFunctionCall",
                details={"error": str(exc)},
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        try:
            sim = await self._backend.simulate(call)
        except _BackendUnreachable as exc:
            fail_closed = state.tier.value in self._fail_closed_tiers
            return StageResult(
                stage=self.name,
                action=Action.BLOCK if fail_closed else Action.PASS,
                severity=3 if fail_closed else 1,
                reason=("simulation backend unreachable; fail-closed for " + state.tier.value)
                if fail_closed
                else "simulation backend unreachable; degraded mode",
                details={"error": str(exc)},
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        sim.suspicious_flags.extend(_post_check(call, sim))
        state.scratch["simulation"] = sim.model_dump()

        if not sim.succeeded:
            return StageResult(
                stage=self.name,
                action=Action.BLOCK,
                severity=3,
                reason=f"simulation reverted: {sim.revert_reason or 'unknown'}",
                details=sim.model_dump(),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        if sim.suspicious_flags:
            return StageResult(
                stage=self.name,
                action=Action.BLOCK if state.tier.value in {"T3", "T4"} else Action.PASS,
                severity=min(5, 2 + len(sim.suspicious_flags)),
                reason="; ".join(sim.suspicious_flags),
                details=sim.model_dump(),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        return StageResult(
            stage=self.name,
            action=Action.PASS,
            reason="simulation clean",
            details=sim.model_dump(),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


def _post_check(call: DAOFunctionCall, sim: SimulationResult) -> list[str]:
    """Heuristics applied after parsing the receipt."""
    flags: list[str] = []
    caller = (call.caller_address or "").lower()

    # Self-dealing: any ERC20 net inflow to the caller wallet is suspicious
    # when the call was supposedly on behalf of someone else.
    for transfer in sim.erc20_transfers:
        if transfer.to_addr.lower() == caller and transfer.amount > 0 and caller:
            flags.append("self-dealing: caller is the recipient of a transfer")
            break

    return flags
