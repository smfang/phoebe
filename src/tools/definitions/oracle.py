"""
Oracle tools — query the SaraOracle contract for evaluation results.

Allows Sara (and other AI agents via the tool interface) to read
on-chain safety evaluation results. This closes the multi-agent loop:
one agent publishes results, others query them.
"""

import hashlib
from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="oracle.query",
    description=(
        "Query the SaraOracle contract for a specific evaluation result "
        "by its evaluation ID. Returns the full result including prompt hash, "
        "category, severity, unsafe flag, and attestation."
    ),
    parameters=[
        ToolParameter(
            name="evaluation_id",
            type="string",
            description="The on-chain evaluation ID (bytes32 hex string)",
        ),
    ],
)
async def oracle_query(
    ctx: ToolContext,
    evaluation_id: str,
) -> dict[str, Any]:
    """Query a single evaluation result from the oracle."""
    oracle = _get_oracle(ctx)
    if not oracle:
        return {"error": "Oracle not configured"}

    try:
        result = await oracle.query_result(evaluation_id)
        return {"evaluation_id": evaluation_id, "result": result}
    except Exception as e:
        return {"error": str(e), "evaluation_id": evaluation_id}


@TOOL_REGISTRY.tool(
    name="oracle.is_unsafe",
    description=(
        "Check if a prompt has been classified as unsafe for a given "
        "safety category on the SaraOracle contract. Returns true/false. "
        "Use this for downstream safety gating decisions."
    ),
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="The prompt text to check (will be SHA-256 hashed)",
        ),
        ToolParameter(
            name="category",
            type="string",
            description="Safety category (e.g. 'hate', 'illicit_activities', 'pii_ip')",
        ),
    ],
)
async def oracle_is_unsafe(
    ctx: ToolContext,
    prompt: str,
    category: str,
) -> dict[str, Any]:
    """Check if a prompt is classified as unsafe on-chain."""
    oracle = _get_oracle(ctx)
    if not oracle:
        return {"error": "Oracle not configured"}

    from src.safety.oracle import CATEGORY_INDEX

    prompt_hash = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
    category_idx = CATEGORY_INDEX.get(category, 0)

    try:
        unsafe = await oracle.is_unsafe(prompt_hash, category_idx)
        severity = await oracle.get_severity(prompt_hash, category_idx)
        return {
            "prompt_hash": prompt_hash,
            "category": category,
            "unsafe": unsafe,
            "severity": severity,
        }
    except Exception as e:
        return {"error": str(e)}


@TOOL_REGISTRY.tool(
    name="oracle.history",
    description=(
        "Get all evaluation IDs for a prompt from the SaraOracle. "
        "Returns a list of evaluation IDs across all categories. "
        "Useful for auditing all safety checks performed on a prompt."
    ),
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="The prompt text (will be SHA-256 hashed)",
        ),
    ],
)
async def oracle_history(
    ctx: ToolContext,
    prompt: str,
) -> dict[str, Any]:
    """Get all oracle evaluation IDs for a prompt."""
    oracle = _get_oracle(ctx)
    if not oracle:
        return {"error": "Oracle not configured"}

    prompt_hash = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

    try:
        history = await oracle.get_history(prompt_hash)
        return {
            "prompt_hash": prompt_hash,
            "evaluation_count": len(history),
            "evaluation_ids": history,
        }
    except Exception as e:
        return {"error": str(e)}


@TOOL_REGISTRY.tool(
    name="oracle.published",
    description=(
        "List all oracle results published during this session. "
        "Returns the local record of all publish transactions."
    ),
    parameters=[],
)
async def oracle_published(ctx: ToolContext) -> dict[str, Any]:
    """List all results published to the oracle in this session."""
    oracle = _get_oracle(ctx)
    if not oracle:
        return {"error": "Oracle not configured"}

    records = oracle.records
    return {
        "count": len(records),
        "records": [
            {
                "evaluation_id": r.evaluation_id,
                "tx_hash": r.tx_hash,
                "prompt_hash": r.prompt_hash[:16] + "...",
                "category": r.category,
                "severity": r.severity,
                "unsafe": r.unsafe,
                "chain": r.chain,
            }
            for r in records
        ],
    }


def _get_oracle(ctx: ToolContext) -> Any:
    """Get the OraclePublisher from the tool context."""
    # The oracle is attached to the context as an optional field.
    # If not configured, tools return a helpful error message.
    return getattr(ctx, "_oracle", None)
