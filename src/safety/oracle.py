"""
PhoebeOracle on-chain client.

Publishes safety evaluation results to the PhoebeOracle contract,
making them available to smart contracts (via view functions) and
AI agents (via events). Follows the same dual-mode pattern as
erc8004.py: direct on-chain transactions or HTTP relayer.

The oracle contract stores:
  - promptHash: SHA-256 of the evaluated prompt
  - category: GA Guard safety category index
  - severity: 1-5 severity rating
  - unsafe: whether the output was classified as unsafe
  - attestation: evaluator signature or TEE quote

Usage:
    oracle = OraclePublisher(
        contract_address="0x...",
        chain="base",
        rpc_url="https://mainnet.base.org",
        private_key="0x...",
    )
    record = await oracle.publish_result(
        prompt_hash="abc123...",
        category=3,
        severity=4,
        unsafe=True,
        attestation=b"...",
    )
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# GA Guard category name → uint8 index mapping (matches contract)
CATEGORY_INDEX: dict[str, int] = {
    "pii_ip": 0,
    "illicit_activities": 1,
    "hate": 2,
    "sexual_content": 3,
    "prompt_security": 4,
    "violence_self_harm": 5,
    "misinformation": 6,
}


# Minimal ABI for PhoebeOracle — only the functions we call
ORACLE_ABI = [
    {
        "name": "publishResult",
        "type": "function",
        "inputs": [
            {"name": "promptHash", "type": "bytes32"},
            {"name": "category", "type": "uint8"},
            {"name": "severity", "type": "uint8"},
            {"name": "unsafe", "type": "bool"},
            {"name": "attestation", "type": "bytes"},
        ],
        "outputs": [{"name": "evaluationId", "type": "bytes32"}],
    },
    {
        "name": "publishResultWithTEE",
        "type": "function",
        "inputs": [
            {"name": "promptHash", "type": "bytes32"},
            {"name": "category", "type": "uint8"},
            {"name": "severity", "type": "uint8"},
            {"name": "unsafe", "type": "bool"},
            {"name": "enclaveHash", "type": "bytes32"},
            {"name": "attestation", "type": "bytes"},
        ],
        "outputs": [{"name": "evaluationId", "type": "bytes32"}],
    },
    {
        "name": "getResult",
        "type": "function",
        "inputs": [{"name": "evaluationId", "type": "bytes32"}],
        "outputs": [
            {
                "name": "result",
                "type": "tuple",
                "components": [
                    {"name": "promptHash", "type": "bytes32"},
                    {"name": "category", "type": "uint8"},
                    {"name": "severity", "type": "uint8"},
                    {"name": "unsafe", "type": "bool"},
                    {"name": "timestamp", "type": "uint64"},
                    {"name": "enclaveHash", "type": "bytes32"},
                    {"name": "attestation", "type": "bytes"},
                ],
            }
        ],
    },
    {
        "name": "isUnsafe",
        "type": "function",
        "inputs": [
            {"name": "promptHash", "type": "bytes32"},
            {"name": "category", "type": "uint8"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "getSeverity",
        "type": "function",
        "inputs": [
            {"name": "promptHash", "type": "bytes32"},
            {"name": "category", "type": "uint8"},
        ],
        "outputs": [{"name": "", "type": "uint8"}],
    },
    {
        "name": "getHistory",
        "type": "function",
        "inputs": [{"name": "promptHash", "type": "bytes32"}],
        "outputs": [{"name": "", "type": "bytes32[]"}],
    },
]


def _to_bytes32(hex_str: str) -> str:
    """Normalize a hex string to a 0x-prefixed bytes32."""
    clean = hex_str.replace("0x", "")
    if len(clean) != 64:
        clean = hashlib.sha256(clean.encode()).hexdigest()
    return f"0x{clean}"


def _chain_id(chain: str) -> int:
    return {
        "base": 8453,
        "ethereum": 1,
        "polygon": 137,
        "arbitrum": 42161,
        "optimism": 10,
        "base-sepolia": 84532,
        "sepolia": 11155111,
    }.get(chain, 8453)


@dataclass
class OracleRecord:
    """A published oracle result."""

    evaluation_id: str
    tx_hash: str
    prompt_hash: str
    category: int
    severity: int
    unsafe: bool
    timestamp: int
    chain: str
    contract_address: str


@dataclass
class OraclePublisher:
    """
    Publishes evaluation results to the PhoebeOracle contract.

    Two modes:
    - Direct: uses eth_account to sign and submit transactions
    - Relayer: posts to an HTTP relayer that handles gas
    """

    contract_address: str
    chain: str = "base"
    rpc_url: str = ""
    relayer_url: str = ""
    publisher_address: str = ""
    private_key: str = ""
    _http: httpx.AsyncClient = field(default_factory=lambda: httpx.AsyncClient(timeout=30.0))
    _records: list[OracleRecord] = field(default_factory=list)

    @property
    def records(self) -> list[OracleRecord]:
        return list(self._records)

    async def publish_result(
        self,
        prompt_hash: str,
        category: int,
        severity: int,
        unsafe: bool,
        attestation: bytes = b"",
        enclave_hash: str | None = None,
    ) -> OracleRecord:
        """
        Publish an evaluation result to the oracle contract.

        Args:
            prompt_hash: SHA-256 hex of the evaluated prompt.
            category: GA Guard category index (0-6).
            severity: Severity rating 1-5.
            unsafe: Whether the output was classified as unsafe.
            attestation: Evaluator signature or TEE attestation bytes.
            enclave_hash: If set, uses publishResultWithTEE for TEE-attested results.

        Returns:
            OracleRecord with the on-chain evaluation ID and tx hash.
        """
        if self.relayer_url:
            return await self._publish_via_relayer(
                prompt_hash, category, severity, unsafe, attestation, enclave_hash,
            )
        return await self._publish_direct(
            prompt_hash, category, severity, unsafe, attestation, enclave_hash,
        )

    async def query_result(self, evaluation_id: str) -> dict[str, Any]:
        """Read a result from the oracle contract."""
        selector = "0x7fea5047"  # getResult(bytes32)
        encoded_id = evaluation_id.replace("0x", "").zfill(64)

        resp = await self._rpc_call(
            self.contract_address, f"{selector}{encoded_id}"
        )
        return {"raw": resp, "evaluation_id": evaluation_id}

    async def is_unsafe(self, prompt_hash: str, category: int) -> bool:
        """Check if a prompt is classified as unsafe for a category."""
        selector = "0xb1a2567e"  # isUnsafe(bytes32,uint8)
        encoded_hash = prompt_hash.replace("0x", "").zfill(64)
        encoded_cat = hex(category)[2:].zfill(64)

        result = await self._rpc_call(
            self.contract_address, f"{selector}{encoded_hash}{encoded_cat}"
        )
        return int(result, 16) == 1 if result and result != "0x" else False

    async def get_severity(self, prompt_hash: str, category: int) -> int:
        """Get severity for a prompt+category from the oracle."""
        selector = "0x3e8d3764"  # getSeverity(bytes32,uint8)
        encoded_hash = prompt_hash.replace("0x", "").zfill(64)
        encoded_cat = hex(category)[2:].zfill(64)

        result = await self._rpc_call(
            self.contract_address, f"{selector}{encoded_hash}{encoded_cat}"
        )
        return int(result, 16) if result and result != "0x" else 0

    async def get_history(self, prompt_hash: str) -> list[str]:
        """Get all evaluation IDs for a prompt."""
        selector = "0x457f4bcc"  # getHistory(bytes32)
        encoded_hash = prompt_hash.replace("0x", "").zfill(64)

        result = await self._rpc_call(
            self.contract_address, f"{selector}{encoded_hash}"
        )
        if not result or result == "0x":
            return []

        # Decode dynamic bytes32 array (offset + length + elements)
        raw = result.replace("0x", "")
        if len(raw) < 128:
            return []
        length = int(raw[64:128], 16)
        ids = []
        for i in range(length):
            start = 128 + i * 64
            ids.append(f"0x{raw[start:start + 64]}")
        return ids

    # ------------------------------------------------------------------
    # Internal: relayer mode
    # ------------------------------------------------------------------

    async def _publish_via_relayer(
        self,
        prompt_hash: str,
        category: int,
        severity: int,
        unsafe: bool,
        attestation: bytes,
        enclave_hash: str | None,
    ) -> OracleRecord:
        payload: dict[str, Any] = {
            "contract": self.contract_address,
            "chain": self.chain,
            "promptHash": _to_bytes32(prompt_hash),
            "category": category,
            "severity": severity,
            "unsafe": unsafe,
            "attestation": f"0x{attestation.hex()}" if attestation else "0x",
        }
        if enclave_hash:
            payload["enclaveHash"] = _to_bytes32(enclave_hash)

        resp = await self._http.post(
            f"{self.relayer_url.rstrip('/')}/oracle/publish",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        record = OracleRecord(
            evaluation_id=data.get("evaluationId", ""),
            tx_hash=data.get("txHash", ""),
            prompt_hash=prompt_hash,
            category=category,
            severity=severity,
            unsafe=unsafe,
            timestamp=int(time.time()),
            chain=self.chain,
            contract_address=self.contract_address,
        )
        self._records.append(record)
        logger.info(
            "Oracle result published via relayer: eval=%s tx=%s",
            record.evaluation_id[:16], record.tx_hash[:16],
        )
        return record

    # ------------------------------------------------------------------
    # Internal: direct on-chain mode
    # ------------------------------------------------------------------

    async def _publish_direct(
        self,
        prompt_hash: str,
        category: int,
        severity: int,
        unsafe: bool,
        attestation: bytes,
        enclave_hash: str | None,
    ) -> OracleRecord:
        if not self.rpc_url:
            raise RuntimeError("No RPC URL configured for direct oracle publishing")
        if not self.private_key:
            raise RuntimeError("No private key configured for direct oracle publishing")

        from eth_account import Account

        account = Account.from_key(self.private_key)

        if enclave_hash:
            call_data = self._encode_publish_with_tee(
                prompt_hash, category, severity, unsafe, enclave_hash, attestation,
            )
        else:
            call_data = self._encode_publish(
                prompt_hash, category, severity, unsafe, attestation,
            )

        # Get nonce
        nonce_resp = await self._http.post(
            self.rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_getTransactionCount",
                "params": [account.address, "latest"],
                "id": 1,
            },
        )
        nonce = int(nonce_resp.json()["result"], 16)

        tx = {
            "to": self.contract_address,
            "data": call_data,
            "nonce": nonce,
            "gas": 250_000,
            "maxFeePerGas": 1_000_000_000,
            "maxPriorityFeePerGas": 100_000_000,
            "chainId": _chain_id(self.chain),
            "type": 2,
        }

        signed = account.sign_transaction(tx)

        send_resp = await self._http.post(
            self.rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_sendRawTransaction",
                "params": [signed.raw_transaction.hex()],
                "id": 2,
            },
        )
        tx_hash = send_resp.json()["result"]

        record = OracleRecord(
            evaluation_id="",  # resolved from tx receipt events
            tx_hash=tx_hash,
            prompt_hash=prompt_hash,
            category=category,
            severity=severity,
            unsafe=unsafe,
            timestamp=int(time.time()),
            chain=self.chain,
            contract_address=self.contract_address,
        )
        self._records.append(record)
        logger.info("Oracle tx submitted: %s", tx_hash)
        return record

    # ------------------------------------------------------------------
    # Internal: ABI encoding
    # ------------------------------------------------------------------

    def _encode_publish(
        self,
        prompt_hash: str,
        category: int,
        severity: int,
        unsafe: bool,
        attestation: bytes,
    ) -> str:
        """ABI-encode publishResult(bytes32,uint8,uint8,bool,bytes)."""
        # selector: keccak256("publishResult(bytes32,uint8,uint8,bool,bytes)")[:4]
        selector = "0x6a627842"
        ph = _to_bytes32(prompt_hash).replace("0x", "")
        cat = hex(category)[2:].zfill(64)
        sev = hex(severity)[2:].zfill(64)
        uns = "1".zfill(64) if unsafe else "0".zfill(64)
        # dynamic bytes offset (5 * 32 = 160 = 0xa0)
        offset = hex(160)[2:].zfill(64)
        # attestation bytes
        att_hex = attestation.hex() if attestation else ""
        att_len = hex(len(attestation))[2:].zfill(64)
        att_padded = att_hex + "0" * ((64 - len(att_hex) % 64) % 64)

        return selector + ph + cat + sev + uns + offset + att_len + att_padded

    def _encode_publish_with_tee(
        self,
        prompt_hash: str,
        category: int,
        severity: int,
        unsafe: bool,
        enclave_hash: str,
        attestation: bytes,
    ) -> str:
        """ABI-encode publishResultWithTEE(bytes32,uint8,uint8,bool,bytes32,bytes)."""
        selector = "0x8b4e3c18"
        ph = _to_bytes32(prompt_hash).replace("0x", "")
        cat = hex(category)[2:].zfill(64)
        sev = hex(severity)[2:].zfill(64)
        uns = "1".zfill(64) if unsafe else "0".zfill(64)
        eh = _to_bytes32(enclave_hash).replace("0x", "")
        # dynamic bytes offset (6 * 32 = 192 = 0xc0)
        offset = hex(192)[2:].zfill(64)
        att_hex = attestation.hex() if attestation else ""
        att_len = hex(len(attestation))[2:].zfill(64)
        att_padded = att_hex + "0" * ((64 - len(att_hex) % 64) % 64)

        return selector + ph + cat + sev + uns + eh + offset + att_len + att_padded

    # ------------------------------------------------------------------
    # Internal: JSON-RPC helper
    # ------------------------------------------------------------------

    async def _rpc_call(self, to: str, data: str) -> str:
        """Execute a read-only eth_call."""
        if not self.rpc_url:
            if self.relayer_url:
                resp = await self._http.post(
                    f"{self.relayer_url.rstrip('/')}/oracle/call",
                    json={"to": to, "data": data},
                )
                resp.raise_for_status()
                return resp.json().get("result", "0x")
            raise RuntimeError("No RPC URL or relayer configured for oracle reads")

        resp = await self._http.post(
            self.rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{"to": to, "data": data}, "latest"],
                "id": 1,
            },
        )
        return resp.json().get("result", "0x")
