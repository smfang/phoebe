"""
Oracle event indexer — caches PhoebeOracle events into a queryable store.

Listens for ResultPublished events from the PhoebeOracle contract and
maintains a local cache (in-memory + optional ClickHouse persistence).
Exposes a REST-like interface for AI agents to query results without
making direct RPC calls.

This is the "thin indexer" layer that makes oracle data agent-friendly:
  - Subscribe to events via websocket or polling
  - Query by promptHash, category, time range
  - Get aggregated stats (total results, category breakdown)

Usage:
    indexer = OracleIndexer(
        rpc_url="wss://mainnet.base.org",
        contract_address="0x...",
    )
    await indexer.start()

    # Query cached results
    results = indexer.query(prompt_hash="abc...")
    stats = indexer.stats()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ResultPublished event signature:
# keccak256("ResultPublished(bytes32,bytes32,uint8,uint8,bool)")
RESULT_PUBLISHED_TOPIC = (
    "0x"  # placeholder — compute from actual contract deployment
    "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
)


@dataclass
class IndexedResult:
    """A cached oracle evaluation result."""

    evaluation_id: str
    prompt_hash: str
    category: int
    severity: int
    unsafe: bool
    block_number: int
    tx_hash: str
    timestamp: int


@dataclass
class OracleIndexer:
    """
    Indexes PhoebeOracle ResultPublished events.

    Supports two modes:
    - Websocket: Real-time event subscription (preferred)
    - Polling: Periodic eth_getLogs polling (fallback)
    """

    rpc_url: str
    contract_address: str
    poll_interval: float = 12.0  # seconds (one block on Base)
    start_block: int = 0
    _http: httpx.AsyncClient = field(
        default_factory=lambda: httpx.AsyncClient(timeout=30.0)
    )
    _results: dict[str, IndexedResult] = field(default_factory=dict)
    _by_prompt: dict[str, list[str]] = field(default_factory=dict)
    _running: bool = False
    _last_block: int = 0

    @property
    def result_count(self) -> int:
        return len(self._results)

    def query(
        self,
        prompt_hash: str | None = None,
        category: int | None = None,
        limit: int = 100,
    ) -> list[IndexedResult]:
        """Query cached results by prompt hash and/or category."""
        results: list[IndexedResult]

        if prompt_hash:
            eval_ids = self._by_prompt.get(prompt_hash, [])
            results = [self._results[eid] for eid in eval_ids if eid in self._results]
        else:
            results = list(self._results.values())

        if category is not None:
            results = [r for r in results if r.category == category]

        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:limit]

    def stats(self) -> dict[str, Any]:
        """Get aggregate stats about indexed results."""
        by_category: dict[int, int] = {}
        unsafe_count = 0

        for r in self._results.values():
            by_category[r.category] = by_category.get(r.category, 0) + 1
            if r.unsafe:
                unsafe_count += 1

        return {
            "total_results": len(self._results),
            "unsafe_count": unsafe_count,
            "safe_count": len(self._results) - unsafe_count,
            "by_category": by_category,
            "last_block": self._last_block,
            "indexer_running": self._running,
        }

    async def start(self) -> None:
        """Start the event indexer (polling mode)."""
        if self._running:
            return

        self._running = True
        self._last_block = self.start_block or await self._get_latest_block()

        logger.info(
            "Oracle indexer started: contract=%s from_block=%d",
            self.contract_address[:16], self._last_block,
        )

        asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop the indexer."""
        self._running = False
        logger.info("Oracle indexer stopped")

    async def backfill(self, from_block: int, to_block: int | None = None) -> int:
        """Backfill historical events from a block range."""
        if to_block is None:
            to_block = await self._get_latest_block()

        count = 0
        # Process in chunks of 10,000 blocks
        chunk_size = 10_000
        current = from_block

        while current <= to_block:
            end = min(current + chunk_size - 1, to_block)
            events = await self._fetch_logs(current, end)
            for event in events:
                self._index_event(event)
                count += 1
            current = end + 1

        logger.info("Backfilled %d events from blocks %d-%d", count, from_block, to_block)
        return count

    # ------------------------------------------------------------------
    # Internal: polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Poll for new events."""
        while self._running:
            try:
                latest = await self._get_latest_block()
                if latest > self._last_block:
                    events = await self._fetch_logs(self._last_block + 1, latest)
                    for event in events:
                        self._index_event(event)
                    self._last_block = latest
            except Exception as e:
                logger.warning("Indexer poll error: %s", e)

            await asyncio.sleep(self.poll_interval)

    async def _fetch_logs(self, from_block: int, to_block: int) -> list[dict]:
        """Fetch ResultPublished logs from the RPC."""
        resp = await self._http.post(
            self.rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_getLogs",
                "params": [
                    {
                        "address": self.contract_address,
                        "fromBlock": hex(from_block),
                        "toBlock": hex(to_block),
                        "topics": [RESULT_PUBLISHED_TOPIC],
                    }
                ],
                "id": 1,
            },
        )
        data = resp.json()
        return data.get("result", [])

    def _index_event(self, log: dict) -> None:
        """Parse and index a single ResultPublished event log."""
        topics = log.get("topics", [])
        if len(topics) < 3:
            return

        evaluation_id = topics[1]  # indexed
        prompt_hash = topics[2]  # indexed

        # Non-indexed data: category (uint8), severity (uint8), unsafe (bool)
        raw_data = log.get("data", "0x").replace("0x", "")
        if len(raw_data) < 192:
            return

        category = int(raw_data[0:64], 16)
        severity = int(raw_data[64:128], 16)
        unsafe = int(raw_data[128:192], 16) == 1

        result = IndexedResult(
            evaluation_id=evaluation_id,
            prompt_hash=prompt_hash,
            category=category,
            severity=severity,
            unsafe=unsafe,
            block_number=int(log.get("blockNumber", "0x0"), 16),
            tx_hash=log.get("transactionHash", ""),
            timestamp=int(time.time()),  # approximate; use block timestamp for precision
        )

        self._results[evaluation_id] = result

        if prompt_hash not in self._by_prompt:
            self._by_prompt[prompt_hash] = []
        if evaluation_id not in self._by_prompt[prompt_hash]:
            self._by_prompt[prompt_hash].append(evaluation_id)

    async def _get_latest_block(self) -> int:
        """Get the latest block number from the RPC."""
        resp = await self._http.post(
            self.rpc_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_blockNumber",
                "params": [],
                "id": 1,
            },
        )
        return int(resp.json()["result"], 16)
