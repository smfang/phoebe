"""
Evaluation metrics for the Indeed policy classification eval.

Computes per-policy and aggregate F1, precision, recall, and confusion
matrices. Results are designed to be comparable with the GA Guard
adversarial evaluation benchmarks:

  - GA Guard (adversarial RL): F1 0.840
  - GPT-4o baseline:           F1 0.623
  - Phoebe (LLM-as-judge):     expected ~0.60-0.65 without fine-tuning
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConfusionMatrix:
    """Binary confusion matrix for a single policy."""

    tp: int = 0  # true positive: predicted violation, actually violation
    fp: int = 0  # false positive: predicted violation, actually compliant
    tn: int = 0  # true negative: predicted compliant, actually compliant
    fn: int = 0  # false negative: predicted compliant, actually violation

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class ClassificationResult:
    """Result of classifying a single sample against one policy."""

    sample_id: int
    policy_id: str
    predicted_unsafe: bool
    ground_truth_unsafe: bool
    severity: int = 0
    explanation: str = ""


@dataclass
class EvalResults:
    """Aggregated evaluation results across all policies."""

    per_policy: dict[str, ConfusionMatrix] = field(default_factory=dict)
    overall: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    results: list[ClassificationResult] = field(default_factory=list)
    total_samples: int = 0
    total_classifications: int = 0

    def add_result(self, result: ClassificationResult) -> None:
        """Add a single classification result and update confusion matrices."""
        self.results.append(result)
        self.total_classifications += 1

        # Per-policy matrix
        if result.policy_id not in self.per_policy:
            self.per_policy[result.policy_id] = ConfusionMatrix()
        cm = self.per_policy[result.policy_id]

        # Overall matrix
        if result.predicted_unsafe and result.ground_truth_unsafe:
            cm.tp += 1
            self.overall.tp += 1
        elif result.predicted_unsafe and not result.ground_truth_unsafe:
            cm.fp += 1
            self.overall.fp += 1
        elif not result.predicted_unsafe and not result.ground_truth_unsafe:
            cm.tn += 1
            self.overall.tn += 1
        else:
            cm.fn += 1
            self.overall.fn += 1

    @property
    def macro_f1(self) -> float:
        """Macro-averaged F1 across all policies."""
        if not self.per_policy:
            return 0.0
        f1s = [cm.f1 for cm in self.per_policy.values()]
        return sum(f1s) / len(f1s)

    @property
    def macro_precision(self) -> float:
        if not self.per_policy:
            return 0.0
        ps = [cm.precision for cm in self.per_policy.values()]
        return sum(ps) / len(ps)

    @property
    def macro_recall(self) -> float:
        if not self.per_policy:
            return 0.0
        rs = [cm.recall for cm in self.per_policy.values()]
        return sum(rs) / len(rs)

    def summary(self) -> str:
        """Human-readable summary of eval results."""
        lines = [
            "=" * 70,
            "EVAL RESULTS — Indeed Policy Classification",
            "=" * 70,
            f"Total samples:         {self.total_samples}",
            f"Total classifications: {self.total_classifications}",
            "",
            f"Overall F1:        {self.overall.f1:.4f}",
            f"Overall Precision: {self.overall.precision:.4f}",
            f"Overall Recall:    {self.overall.recall:.4f}",
            f"Overall Accuracy:  {self.overall.accuracy:.4f}",
            "",
            f"Macro F1:          {self.macro_f1:.4f}",
            f"Macro Precision:   {self.macro_precision:.4f}",
            f"Macro Recall:      {self.macro_recall:.4f}",
            "",
            "-" * 70,
            f"{'Policy':<25} {'F1':>8} {'Prec':>8} {'Recall':>8} "
            f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}",
            "-" * 70,
        ]

        for policy_id in sorted(self.per_policy.keys()):
            cm = self.per_policy[policy_id]
            lines.append(
                f"{policy_id:<25} {cm.f1:>8.4f} {cm.precision:>8.4f} "
                f"{cm.recall:>8.4f} {cm.tp:>5} {cm.fp:>5} {cm.tn:>5} {cm.fn:>5}"
            )

        lines.append("-" * 70)

        # GA Guard comparison
        lines.extend([
            "",
            "Comparison benchmarks:",
            f"  GA Guard (adversarial RL):  F1 0.8400",
            f"  GPT-4o baseline:            F1 0.6230",
            f"  Phoebe (this run):          F1 {self.overall.f1:.4f}",
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializable dict for JSON output."""
        return {
            "total_samples": self.total_samples,
            "total_classifications": self.total_classifications,
            "overall": self.overall.to_dict(),
            "macro_f1": round(self.macro_f1, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "per_policy": {
                pid: cm.to_dict() for pid, cm in sorted(self.per_policy.items())
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def false_negatives(self) -> list[ClassificationResult]:
        """Return all false negatives (missed violations) for analysis."""
        return [
            r for r in self.results
            if r.ground_truth_unsafe and not r.predicted_unsafe
        ]

    def false_positives(self) -> list[ClassificationResult]:
        """Return all false positives (over-flagged compliant posts) for analysis."""
        return [
            r for r in self.results
            if not r.ground_truth_unsafe and r.predicted_unsafe
        ]
