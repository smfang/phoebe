"""
Tests for the Indeed policy eval harness.

All tests run offline with mocked classifier calls — no API key needed.
"""

import csv
import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.eval.dataset import EvalDataset, EvalSample, load_csv, load_dataset
from src.eval.indeed_policies import (
    ALL_INDEED_POLICY_IDS,
    INDEED_POLICIES,
    get_all_ga_guard_categories,
    get_ga_guard_categories_for_policy,
)
from src.eval.metrics import ClassificationResult, ConfusionMatrix, EvalResults
from src.eval.runner import run_eval
from src.eval.synthetic import generate_synthetic_violations
from src.safety.classifier import SafetyClassifier


# ---------------------------------------------------------------------------
# Indeed policies
# ---------------------------------------------------------------------------


class TestIndeedPolicies:
    def test_all_10_policies_defined(self):
        assert len(INDEED_POLICIES) == 10
        assert len(ALL_INDEED_POLICY_IDS) == 10

    def test_each_policy_has_ga_guard_mapping(self):
        for pid, policy in INDEED_POLICIES.items():
            assert len(policy.ga_guard_categories) >= 1, (
                f"Policy {pid} has no GA Guard category mapping"
            )

    def test_get_ga_guard_categories(self):
        cats = get_ga_guard_categories_for_policy("discrimination")
        assert "hate" in cats

        cats = get_ga_guard_categories_for_policy("pii_harvesting")
        assert "pii_ip" in cats

    def test_get_ga_guard_categories_unknown(self):
        assert get_ga_guard_categories_for_policy("nonexistent") == []

    def test_all_ga_guard_categories_covered(self):
        all_cats = get_all_ga_guard_categories()
        # Indeed policies should map to at least 5 of the 7 GA Guard categories
        assert len(all_cats) >= 5

    def test_each_policy_has_examples(self):
        for pid, policy in INDEED_POLICIES.items():
            assert len(policy.examples_block) >= 1, f"{pid} missing block examples"
            assert len(policy.examples_allow) >= 1, f"{pid} missing allow examples"


# ---------------------------------------------------------------------------
# Confusion matrix / metrics
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_perfect_classifier(self):
        cm = ConfusionMatrix(tp=50, fp=0, tn=50, fn=0)
        assert cm.precision == 1.0
        assert cm.recall == 1.0
        assert cm.f1 == 1.0
        assert cm.accuracy == 1.0

    def test_all_false_positives(self):
        cm = ConfusionMatrix(tp=0, fp=50, tn=50, fn=0)
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.f1 == 0.0

    def test_all_false_negatives(self):
        cm = ConfusionMatrix(tp=0, fp=0, tn=50, fn=50)
        assert cm.precision == 0.0
        assert cm.recall == 0.0

    def test_realistic_classifier(self):
        # Simulating ~F1 0.62 (similar to GPT-4o baseline)
        cm = ConfusionMatrix(tp=31, fp=12, tn=38, fn=19)
        assert 0.55 < cm.f1 < 0.70
        assert 0.60 < cm.precision < 0.80
        assert 0.55 < cm.recall < 0.70

    def test_empty_matrix(self):
        cm = ConfusionMatrix()
        assert cm.f1 == 0.0
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.total == 0

    def test_to_dict(self):
        cm = ConfusionMatrix(tp=10, fp=2, tn=80, fn=8)
        d = cm.to_dict()
        assert d["tp"] == 10
        assert d["fp"] == 2
        assert "f1" in d
        assert "precision" in d


class TestEvalResults:
    def test_add_results_and_compute(self):
        results = EvalResults(total_samples=4)

        # True positive
        results.add_result(ClassificationResult(
            sample_id=0, policy_id="discrimination",
            predicted_unsafe=True, ground_truth_unsafe=True,
        ))
        # True negative
        results.add_result(ClassificationResult(
            sample_id=1, policy_id="discrimination",
            predicted_unsafe=False, ground_truth_unsafe=False,
        ))
        # False positive
        results.add_result(ClassificationResult(
            sample_id=2, policy_id="discrimination",
            predicted_unsafe=True, ground_truth_unsafe=False,
        ))
        # False negative
        results.add_result(ClassificationResult(
            sample_id=3, policy_id="discrimination",
            predicted_unsafe=False, ground_truth_unsafe=True,
        ))

        assert results.overall.tp == 1
        assert results.overall.fp == 1
        assert results.overall.tn == 1
        assert results.overall.fn == 1
        assert results.total_classifications == 4

        cm = results.per_policy["discrimination"]
        assert cm.tp == 1
        assert cm.f1 == pytest.approx(0.5)

    def test_summary_and_json(self):
        results = EvalResults(total_samples=2)
        results.add_result(ClassificationResult(
            sample_id=0, policy_id="hate_speech",
            predicted_unsafe=True, ground_truth_unsafe=True,
        ))
        results.add_result(ClassificationResult(
            sample_id=1, policy_id="hate_speech",
            predicted_unsafe=False, ground_truth_unsafe=False,
        ))

        summary = results.summary()
        assert "EVAL RESULTS" in summary
        assert "hate_speech" in summary

        j = json.loads(results.to_json())
        assert j["total_samples"] == 2
        assert "per_policy" in j
        assert "hate_speech" in j["per_policy"]

    def test_false_negatives_and_positives(self):
        results = EvalResults(total_samples=2)
        results.add_result(ClassificationResult(
            sample_id=0, policy_id="scam_postings",
            predicted_unsafe=False, ground_truth_unsafe=True,
            explanation="missed scam",
        ))
        results.add_result(ClassificationResult(
            sample_id=1, policy_id="scam_postings",
            predicted_unsafe=True, ground_truth_unsafe=False,
            explanation="over-flagged",
        ))

        fn = results.false_negatives()
        assert len(fn) == 1
        assert fn[0].explanation == "missed scam"

        fp = results.false_positives()
        assert len(fp) == 1
        assert fp[0].explanation == "over-flagged"

    def test_macro_f1(self):
        results = EvalResults(total_samples=4)

        # Policy A: perfect
        results.add_result(ClassificationResult(
            sample_id=0, policy_id="a",
            predicted_unsafe=True, ground_truth_unsafe=True,
        ))
        results.add_result(ClassificationResult(
            sample_id=1, policy_id="a",
            predicted_unsafe=False, ground_truth_unsafe=False,
        ))

        # Policy B: all wrong
        results.add_result(ClassificationResult(
            sample_id=2, policy_id="b",
            predicted_unsafe=False, ground_truth_unsafe=True,
        ))
        results.add_result(ClassificationResult(
            sample_id=3, policy_id="b",
            predicted_unsafe=True, ground_truth_unsafe=False,
        ))

        # Macro F1 = avg(1.0, 0.0) = 0.5
        assert results.macro_f1 == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestDataset:
    def test_load_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "job_text,ground_truth\n"
            "Great software engineer role,compliant\n"
            "Young females only apply,discrimination\n"
            '"Earn $10k/week, no experience!","misleading_info,scam_postings"\n'
        )

        samples = load_csv(csv_path, source="test")
        assert len(samples) == 3
        assert samples[0].is_compliant is True
        assert samples[1].ground_truth_policies == ["discrimination"]
        assert samples[1].is_compliant is False
        assert "misleading_info" in samples[2].ground_truth_policies
        assert "scam_postings" in samples[2].ground_truth_policies

    def test_load_csv_fallback_columns(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "description,ground_truth\n"
            "A valid job posting,compliant\n"
        )

        samples = load_csv(csv_path, source="test")
        assert len(samples) == 1
        assert samples[0].job_text == "A valid job posting"

    def test_load_csv_missing_file(self, tmp_path):
        samples = load_csv(tmp_path / "nonexistent.csv")
        assert samples == []

    def test_dataset_sampling(self):
        samples = [
            EvalSample(job_text=f"job {i}", ground_truth_policies=[], is_compliant=True, row_id=i)
            for i in range(90)
        ] + [
            EvalSample(
                job_text=f"violation {i}",
                ground_truth_policies=["discrimination"],
                is_compliant=False,
                row_id=90 + i,
            )
            for i in range(10)
        ]

        ds = EvalDataset(samples=samples)
        assert ds.total == 100
        assert ds.compliance_rate == pytest.approx(0.9)

        sampled = ds.sample(20, stratify=True)
        assert sampled.total == 20
        # Stratified: should have ~2 violations
        assert sampled.violation_count >= 1

    def test_dataset_summary(self):
        ds = EvalDataset(samples=[
            EvalSample(job_text="x", ground_truth_policies=[], is_compliant=True),
        ])
        assert "1 samples" in ds.summary()

    def test_load_dataset_empty_dir_falls_back_to_seed(self, tmp_path):
        ds = load_dataset(data_dir=tmp_path)
        # Falls back to the built-in seed dataset (30 examples)
        assert ds.total == 30
        assert ds.violation_count > 0
        assert ds.compliant_count > 0

    def test_load_dataset_empty_dir_no_seed(self, tmp_path):
        ds = load_dataset(data_dir=tmp_path, include_seed=False)
        assert ds.total == 0


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------


class TestSyntheticGeneration:
    def test_generate_violations(self):
        rows = generate_synthetic_violations(n_per_policy=5, seed=42)
        # 10 policies × 5 = 50
        assert len(rows) == 50
        assert all("job_text" in r for r in rows)
        assert all("ground_truth" in r for r in rows)

    def test_generate_writes_csv(self, tmp_path):
        out_path = tmp_path / "synthetic.csv"
        rows = generate_synthetic_violations(
            n_per_policy=3, seed=42, output_path=out_path,
        )
        assert out_path.exists()

        # Read back and verify
        with open(out_path) as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)
        assert len(read_rows) == 30

    def test_deterministic_with_seed(self):
        rows_a = generate_synthetic_violations(n_per_policy=10, seed=123)
        rows_b = generate_synthetic_violations(n_per_policy=10, seed=123)
        assert [r["job_text"] for r in rows_a] == [r["job_text"] for r in rows_b]

    def test_all_policies_represented(self):
        rows = generate_synthetic_violations(n_per_policy=1, seed=42)
        policies_seen = {r["ground_truth"] for r in rows}
        assert policies_seen == set(ALL_INDEED_POLICY_IDS)


# ---------------------------------------------------------------------------
# Runner (mocked classifier)
# ---------------------------------------------------------------------------


class TestEvalRunner:
    @pytest.fixture
    def mock_classifier(self):
        classifier = SafetyClassifier(api_key="test-key")
        return classifier

    @pytest.mark.asyncio
    async def test_run_eval_with_mock(self, mock_classifier):
        dataset = EvalDataset(samples=[
            EvalSample(
                job_text="Young females only — apply now!",
                ground_truth_policies=["discrimination"],
                is_compliant=False,
                row_id=0,
            ),
            EvalSample(
                job_text="Software engineer, equal opportunity employer",
                ground_truth_policies=[],
                is_compliant=True,
                row_id=1,
            ),
        ])

        # Mock the classifier to return unsafe for the first, safe for second
        call_count = 0

        async def mock_classify(prompt, model_output, category):
            nonlocal call_count
            call_count += 1
            if "Young females" in model_output:
                return {
                    "unsafe": True,
                    "severity": 3,
                    "policy_category": "hate",
                    "matched_block_rule": "discrimination",
                    "explanation": "Discriminatory language",
                }
            return {
                "unsafe": False,
                "severity": 0,
                "policy_category": "none",
                "matched_block_rule": "",
                "explanation": "Compliant posting",
            }

        mock_classifier.classify = mock_classify

        results = await run_eval(
            classifier=mock_classifier,
            dataset=dataset,
            policies=["discrimination"],
            concurrency=2,
        )

        assert results.total_samples == 2
        assert results.total_classifications == 2
        # The mock correctly identifies the violation
        assert results.overall.tp == 1
        assert results.overall.tn == 1
        assert results.overall.f1 == 1.0

    @pytest.mark.asyncio
    async def test_run_eval_multiple_policies(self, mock_classifier):
        dataset = EvalDataset(samples=[
            EvalSample(
                job_text="Send SSN to apply",
                ground_truth_policies=["pii_harvesting"],
                is_compliant=False,
                row_id=0,
            ),
        ])

        async def mock_classify(prompt, model_output, category):
            if "PII" in prompt:
                return {"unsafe": True, "severity": 4, "explanation": "PII"}
            return {"unsafe": False, "severity": 0, "explanation": "OK"}

        mock_classifier.classify = mock_classify

        results = await run_eval(
            classifier=mock_classifier,
            dataset=dataset,
            policies=["pii_harvesting", "discrimination"],
            concurrency=2,
        )

        assert results.total_classifications == 2
        assert "pii_harvesting" in results.per_policy
        assert "discrimination" in results.per_policy
