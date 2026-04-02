"""
Safety RL Training Data Pipeline.

Pulls safety-related datasets, normalises them into DPO preference pairs
(chosen = safe response, rejected = unsafe response), and writes the merged
corpus to disk as a HuggingFace Dataset.

Supported sources
-----------------
- **R-Judge**                 (``Kwan-Ho/R-Judge``)
- **PKU-SafeRLHF**           (``PKU-Alignment/PKU-SafeRLHF``)
- **HarmBench**               (``harmbench/HarmBench``)
- **BeaverTails**             (``PKU-Alignment/BeaverTails``)
- **CrAI-SafeFuncCall**       (``SentientAGI/crypto-agent-safe-function-calling``)

Each adapter converts its source schema into a common
``{"prompt", "chosen", "rejected"}`` format suitable for TRL's ``DPOTrainer``.
"""

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from src.config import CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default dataset identifiers (can be overridden via CLI / env)
# ---------------------------------------------------------------------------

DATASET_SOURCES: dict[str, str] = {
    "rjudge": "Kwan-Ho/R-Judge",
    "pku_saferlhf": "PKU-Alignment/PKU-SafeRLHF",
    "harmbench": "harmbench/HarmBench",
    "beavertails": "PKU-Alignment/BeaverTails",
    "crypto_safefunc": "SentientAGI/crypto-agent-safe-function-calling",
}


# ---------------------------------------------------------------------------
# Per-source adapters
# ---------------------------------------------------------------------------


def _adapt_rjudge(ds: Dataset) -> list[dict[str, str]]:
    """R-Judge rows contain a prompt, a safe answer, and an unsafe answer."""
    pairs: list[dict[str, str]] = []
    for row in ds:
        prompt = row.get("prompt") or row.get("question") or ""
        safe = row.get("safe_response") or row.get("chosen") or ""
        unsafe = row.get("unsafe_response") or row.get("rejected") or ""
        if not prompt or not safe or not unsafe:
            # Fall back to generic two-response layout
            responses = row.get("responses") or []
            labels = row.get("labels") or row.get("is_safe") or []
            if len(responses) >= 2 and len(labels) >= 2:
                prompt = prompt or row.get("input", "")
                idx_safe = next((i for i, l in enumerate(labels) if l), 0)
                idx_unsafe = next((i for i, l in enumerate(labels) if not l), 1)
                safe = responses[idx_safe]
                unsafe = responses[idx_unsafe]
            else:
                continue
        pairs.append({"prompt": prompt.strip(), "chosen": safe.strip(), "rejected": unsafe.strip()})
    return pairs


def _adapt_pku_saferlhf(ds: Dataset) -> list[dict[str, str]]:
    """PKU-SafeRLHF provides response pairs with ``is_response_0_safe`` flags."""
    pairs: list[dict[str, str]] = []
    for row in ds:
        prompt = row.get("prompt", "")
        r0 = row.get("response_0", "")
        r1 = row.get("response_1", "")
        safe_0 = row.get("is_response_0_safe", True)
        safe_1 = row.get("is_response_1_safe", True)

        if safe_0 == safe_1:
            # Both same safety — use the ``safer`` field if present
            safer = row.get("safer", None)
            if safer == 0:
                chosen, rejected = r0, r1
            elif safer == 1:
                chosen, rejected = r1, r0
            else:
                continue
        elif safe_0 and not safe_1:
            chosen, rejected = r0, r1
        else:
            chosen, rejected = r1, r0

        if prompt and chosen and rejected:
            pairs.append({"prompt": prompt.strip(), "chosen": chosen.strip(), "rejected": rejected.strip()})
    return pairs


def _adapt_harmbench(ds: Dataset) -> list[dict[str, str]]:
    """
    HarmBench rows contain harmful behaviour descriptions.

    We treat the harmful completion as ``rejected`` and synthesise a minimal
    safe refusal as ``chosen`` (the model-under-training should learn to
    refuse these prompts).
    """
    pairs: list[dict[str, str]] = []
    refusal = (
        "I'm sorry, but I can't assist with that request. "
        "It involves content that could be harmful or dangerous. "
        "I'd be happy to help you with something else."
    )
    for row in ds:
        prompt = row.get("prompt") or row.get("behavior") or row.get("goal") or ""
        harmful = row.get("target") or row.get("completion") or row.get("response") or ""
        if not prompt:
            continue
        # If no harmful completion, generate a short placeholder
        if not harmful:
            harmful = f"Sure, here is how to {prompt.lower().rstrip('.')}..."
        pairs.append({"prompt": prompt.strip(), "chosen": refusal, "rejected": harmful.strip()})
    return pairs


def _adapt_beavertails(ds: Dataset) -> list[dict[str, str]]:
    """BeaverTails labels each response as safe/unsafe per category."""
    pairs: list[dict[str, str]] = []
    refusal = (
        "I'm not able to help with that request as it involves "
        "potentially harmful content. Let me know if there's "
        "something else I can help you with."
    )
    for row in ds:
        prompt = row.get("prompt", "")
        response = row.get("response", "")
        is_safe = row.get("is_safe", True)
        if not prompt or not response:
            continue
        if not is_safe:
            pairs.append({"prompt": prompt.strip(), "chosen": refusal, "rejected": response.strip()})
        # Safe rows aren't directly useful as DPO pairs — skip
    return pairs


def _adapt_crypto_safefunc(ds: Dataset) -> list[dict[str, str]]:
    """
    SentientAGI/crypto-agent-safe-function-calling — memory-injection attacks
    against crypto/blockchain function-calling agents.

    Each row has:
    - ``memory``: agent memory (may contain injected malicious payload)
    - ``thinking_trajectory``: safe reasoning chain for the query
    - ``answers``: list of correct (safe) tool calls
    - ``attack``: attack type (e.g. "memory injection") or empty for benign
    - ``domain``: task domain (e.g. "crypto")

    For adversarial rows (attack != ""), the model should follow the safe
    ``thinking_trajectory`` + ``answers`` (chosen) rather than obeying the
    injected instructions in memory (rejected).

    For benign rows, the correct tool call with safe reasoning serves as a
    positive example; we synthesise a naive response that blindly trusts
    memory as rejected.
    """
    pairs: list[dict[str, str]] = []
    for row in ds:
        memory = row.get("memory", "")
        thinking = row.get("thinking_trajectory", "")
        answers_raw = row.get("answers", [])
        attack = row.get("attack", "")
        query = row.get("query") or row.get("prompt") or row.get("instruction") or ""

        # Format the safe tool calls as a readable string
        if isinstance(answers_raw, str):
            try:
                answers_raw = json.loads(answers_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        if isinstance(answers_raw, list):
            safe_calls = json.dumps(answers_raw, indent=2)
        else:
            safe_calls = str(answers_raw)

        # Build the prompt: the query + memory context the agent sees
        if query:
            prompt = f"Query: {query}\n\nAgent Memory:\n{memory}" if memory else query
        elif memory:
            prompt = f"Agent Memory:\n{memory}"
        else:
            continue

        # Build chosen: safe reasoning + correct tool calls
        chosen_parts = []
        if thinking:
            chosen_parts.append(f"Reasoning: {thinking}")
        chosen_parts.append(f"Tool calls: {safe_calls}")
        chosen = "\n".join(chosen_parts)

        if attack:
            # Adversarial row — the rejected response obeys the injection
            rejected = (
                f"Based on the memory context, I'll execute the requested operation.\n"
                f"Tool calls: {memory[:500]}"
            )
        else:
            # Benign row — synthesise a naive rejection that skips reasoning
            rejected = (
                f"Executing without verification.\n"
                f"Tool calls: {safe_calls}"
            )
            # Only include benign rows if we have thinking (otherwise the
            # pair is too similar to be useful for DPO)
            if not thinking:
                continue

        if chosen.strip() and rejected.strip() and prompt.strip():
            pairs.append({
                "prompt": prompt.strip(),
                "chosen": chosen.strip(),
                "rejected": rejected.strip(),
            })

    return pairs


# Adapter registry
_ADAPTERS: dict[str, Any] = {
    "rjudge": _adapt_rjudge,
    "pku_saferlhf": _adapt_pku_saferlhf,
    "harmbench": _adapt_harmbench,
    "beavertails": _adapt_beavertails,
    "crypto_safefunc": _adapt_crypto_safefunc,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def pull_and_format(
    sources: list[str] | None = None,
    max_per_source: int = 10_000,
    output_dir: str = "data/safety_rl",
    split: str = "train",
) -> DatasetDict:
    """
    Pull datasets, adapt to DPO format, merge, and save to disk.

    Parameters
    ----------
    sources
        Which dataset keys to include.  ``None`` means all four.
    max_per_source
        Cap per source to keep the corpus balanced.
    output_dir
        Directory to write the merged HuggingFace dataset.
    split
        Which split to load from each source.

    Returns
    -------
    DatasetDict
        ``{"train": Dataset, "test": Dataset}`` with 90/10 split.
    """
    sources = sources or list(DATASET_SOURCES.keys())
    all_pairs: list[dict[str, str]] = []

    for key in sources:
        hf_id = DATASET_SOURCES.get(key)
        if not hf_id:
            logger.warning("Unknown source %s — skipping", key)
            continue

        adapter = _ADAPTERS.get(key)
        if not adapter:
            logger.warning("No adapter for %s — skipping", key)
            continue

        logger.info("Pulling %s from %s (split=%s) ...", key, hf_id, split)
        try:
            ds = load_dataset(hf_id, split=split)
        except Exception:
            # Some datasets only have a default split or use different names
            try:
                ds = load_dataset(hf_id)
                if isinstance(ds, DatasetDict):
                    ds = ds[list(ds.keys())[0]]
            except Exception as e:
                logger.error("Failed to load %s: %s", hf_id, e)
                continue

        pairs = adapter(ds)
        if max_per_source and len(pairs) > max_per_source:
            pairs = pairs[:max_per_source]

        logger.info("  %s → %d DPO pairs", key, len(pairs))
        all_pairs.extend(pairs)

    if not all_pairs:
        raise ValueError("No DPO pairs produced. Check dataset availability.")

    logger.info("Total DPO pairs: %d", len(all_pairs))

    merged = Dataset.from_list(all_pairs)
    ds_dict = merged.train_test_split(test_size=0.1, seed=42)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(out))
    logger.info("Saved to %s  (train=%d, test=%d)", out, len(ds_dict["train"]), len(ds_dict["test"]))

    return ds_dict
