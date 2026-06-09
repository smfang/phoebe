"""
test_dpo_dataset.py

Tests for the DPO preference pair dataset:
  - to_training_format() produces correct keys
  - chosen responses contain <thinking> blocks for all template pairs
  - filter_by_dao_category() works
  - filter_by_tactic() works
  - save_splits() creates all four JSON files
  - load() round-trips correctly
  - get_stats() reports with_cot count correctly
"""

import json
import pytest
from pathlib import Path

from src.data.dpo_dataset import (
    ATLASTactic, DPODataset, DPOPreferencePair, RiskTier, RoutingContext
)
from src.data.dpo_templates import get_all_template_pairs


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_pair() -> DPOPreferencePair:
    return DPOPreferencePair(
        pair_id="test-pair-1",
        chosen="<thinking>\nStep 1: Analysis.\n</thinking>\nDECISION: violation",
        rejected="DECISION: clean",
        atlas_tactic_label=ATLASTactic.IMPACT,
        routing_context=RoutingContext(
            query_text="Test query",
            model_id="sara-v2",
            task_type="test",
            risk_tier=RiskTier.RED,
            domain="finance",
        ),
        severity=5,
        dao_category="treasury_manipulation",
        thinking_trace="<thinking>\nStep 1: Analysis.\n</thinking>",
        source="synthetic",
    )


@pytest.fixture
def template_pairs():
    return get_all_template_pairs()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_to_training_format_keys(sample_pair):
    """to_training_format() must produce prompt, chosen, rejected keys."""
    result = sample_pair.to_training_format()
    assert "prompt" in result
    assert "chosen" in result
    assert "rejected" in result
    assert isinstance(result["prompt"], str)
    assert isinstance(result["chosen"], str)
    assert isinstance(result["rejected"], str)


def test_template_pairs_chosen_contains_thinking(template_pairs):
    """All 18 template pairs must have <thinking> in their chosen response."""
    assert len(template_pairs) == 18, f"Expected 18 pairs, got {len(template_pairs)}"
    for pair in template_pairs:
        assert "<thinking>" in pair.chosen, (
            f"Pair {pair.pair_id} chosen does not contain <thinking> block.\n"
            f"Category: {pair.dao_category}"
        )
        assert "</thinking>" in pair.chosen, (
            f"Pair {pair.pair_id} chosen is missing closing </thinking> tag."
        )


def test_filter_by_dao_category(template_pairs):
    """filter_by_dao_category() must return only matching category."""
    dataset = DPODataset(template_pairs)
    filtered = dataset.filter_by_dao_category("treasury_manipulation")
    assert len(filtered) > 0
    for pair in filtered:
        assert pair.dao_category == "treasury_manipulation"
    # Non-existent category returns empty
    empty = dataset.filter_by_dao_category("nonexistent_category")
    assert len(empty) == 0


def test_filter_by_tactic(template_pairs):
    """filter_by_tactic() must return only matching tactic."""
    dataset = DPODataset(template_pairs)
    filtered = dataset.filter_by_tactic(ATLASTactic.IMPACT)
    assert len(filtered) > 0
    for pair in filtered:
        assert pair.atlas_tactic_label == ATLASTactic.IMPACT
    # Model evasion pairs exist too
    evasion = dataset.filter_by_tactic(ATLASTactic.MODEL_EVASION)
    assert len(evasion) > 0


def test_save_splits_creates_all_files(tmp_path):
    """save_splits() must create train.json, val.json, test.json, full.json."""
    from src.data.dpo_loader import save_splits
    output_dir = str(tmp_path / "dpo")
    paths = save_splits(output_dir=output_dir, augmentation_factor=2)

    for name in ("train", "val", "test", "full"):
        assert name in paths
        assert Path(paths[name]).exists(), f"{name}.json not created"
        # Verify valid JSON
        with open(paths[name]) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0, f"{name}.json is empty"


def test_load_round_trips(tmp_path, template_pairs):
    """DPODataset.load() round-trips correctly — same count and thinking_trace."""
    dataset = DPODataset(template_pairs)
    save_path = str(tmp_path / "test_dataset.json")
    dataset.save(save_path)

    loaded = DPODataset.load(save_path)
    assert len(loaded) == len(dataset)

    for orig, loaded_pair in zip(dataset.pairs, loaded.pairs):
        assert orig.pair_id == loaded_pair.pair_id
        assert orig.thinking_trace == loaded_pair.thinking_trace
        assert orig.dao_category == loaded_pair.dao_category
        assert orig.atlas_tactic_label == loaded_pair.atlas_tactic_label


def test_get_stats_with_cot_count(template_pairs):
    """get_stats() must correctly count pairs with thinking_trace."""
    # Use a copy to avoid mutating the fixture list
    pairs_copy = list(template_pairs)
    dataset = DPODataset(pairs_copy)
    stats = dataset.get_stats()

    # All 18 template pairs have thinking_trace
    n_base = 18
    assert stats["with_cot"] == n_base
    assert stats["total_pairs"] == n_base

    # Add a pair without CoT
    no_cot_pair = DPOPreferencePair(
        pair_id="no-cot",
        chosen="DECISION: violation",
        rejected="DECISION: clean",
        atlas_tactic_label=ATLASTactic.IMPACT,
        routing_context=RoutingContext(
            query_text="Q", model_id="m", task_type="t",
            risk_tier=RiskTier.RED, domain="finance",
        ),
        severity=4,
        thinking_trace=None,  # No CoT
    )
    dataset.add(no_cot_pair)
    new_stats = dataset.get_stats()
    assert new_stats["with_cot"] == n_base  # CoT count unchanged
    assert new_stats["total_pairs"] == n_base + 1


def test_dataset_length_and_indexing(template_pairs):
    """DPODataset supports len() and indexing."""
    dataset = DPODataset(template_pairs)
    assert len(dataset) == 18
    first = dataset[0]
    assert isinstance(first, DPOPreferencePair)
