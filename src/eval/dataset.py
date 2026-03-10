"""
Dataset loader and sampler for the Indeed job postings eval dataset.

Data sources (in priority order):
    1. data/indeed/job_postings.csv       — user-provided labeled dataset
    2. data/indeed/synthetic_violations.csv — generated violations
    3. src/eval/fixtures/seed_dataset.csv  — 30 hand-curated examples (ships with repo)

The seed dataset provides a ready-to-use baseline so you can run the eval
without downloading anything. For serious benchmarking, add your own
labeled data to data/indeed/.

CSV schema (minimum required columns):
    - job_text: str           (the job posting content — title + description)
    - ground_truth: str       (comma-separated Indeed policy IDs that are violated,
                               or "compliant" if clean)
"""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/indeed")
POSTINGS_FILE = "job_postings.csv"
SYNTHETIC_FILE = "synthetic_violations.csv"
SEED_DATASET = Path(__file__).parent / "fixtures" / "seed_dataset.csv"


@dataclass
class EvalSample:
    """A single eval example."""

    job_text: str
    ground_truth_policies: list[str]
    is_compliant: bool
    source: str = "kaggle"  # "kaggle" or "synthetic"
    row_id: int = 0


@dataclass
class EvalDataset:
    """Loaded eval dataset with basic stats."""

    samples: list[EvalSample] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def compliant_count(self) -> int:
        return sum(1 for s in self.samples if s.is_compliant)

    @property
    def violation_count(self) -> int:
        return sum(1 for s in self.samples if not s.is_compliant)

    @property
    def compliance_rate(self) -> float:
        return self.compliant_count / self.total if self.total else 0.0

    def sample(self, n: int, seed: int = 42, stratify: bool = True) -> EvalDataset:
        """
        Sample n examples from the dataset.

        If stratify=True, maintains the violation/compliant ratio.
        """
        rng = random.Random(seed)

        if n >= self.total:
            return self

        if not stratify:
            sampled = rng.sample(self.samples, n)
            return EvalDataset(samples=sampled)

        # Stratified sampling
        compliant = [s for s in self.samples if s.is_compliant]
        violations = [s for s in self.samples if not s.is_compliant]

        if not violations:
            return EvalDataset(samples=rng.sample(compliant, min(n, len(compliant))))

        violation_ratio = len(violations) / self.total
        n_violations = max(1, int(n * violation_ratio))
        n_compliant = n - n_violations

        sampled_violations = rng.sample(violations, min(n_violations, len(violations)))
        sampled_compliant = rng.sample(compliant, min(n_compliant, len(compliant)))

        result = sampled_violations + sampled_compliant
        rng.shuffle(result)
        return EvalDataset(samples=result)

    def summary(self) -> str:
        return (
            f"EvalDataset: {self.total} samples "
            f"({self.compliant_count} compliant, {self.violation_count} violations, "
            f"{self.compliance_rate:.1%} compliance rate)"
        )


def _parse_ground_truth(value: str) -> tuple[list[str], bool]:
    """Parse ground_truth column into (policy_ids, is_compliant)."""
    value = value.strip().lower()
    if not value or value in ("compliant", "clean", "none", "safe"):
        return [], True
    policies = [p.strip() for p in value.split(",") if p.strip()]
    return policies, len(policies) == 0


def load_csv(path: Path, source: str = "kaggle") -> list[EvalSample]:
    """Load a single CSV file into EvalSamples."""
    if not path.exists():
        logger.warning("Dataset file not found: %s", path)
        return []

    samples: list[EvalSample] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate required columns
        if reader.fieldnames is None:
            logger.error("Empty CSV: %s", path)
            return []

        has_job_text = "job_text" in reader.fieldnames
        has_ground_truth = "ground_truth" in reader.fieldnames

        if not has_job_text:
            # Try fallback columns
            text_col = None
            for candidate in ["description", "job_description", "posting_text", "text"]:
                if candidate in reader.fieldnames:
                    text_col = candidate
                    break
            if not text_col:
                logger.error(
                    "CSV %s has no job_text or known text column. "
                    "Columns found: %s",
                    path,
                    reader.fieldnames,
                )
                return []
        else:
            text_col = "job_text"

        for i, row in enumerate(reader):
            job_text = row.get(text_col, "").strip()
            if not job_text:
                continue

            if has_ground_truth:
                policies, is_compliant = _parse_ground_truth(
                    row.get("ground_truth", "compliant")
                )
            else:
                # No labels — assume compliant (user must add labels)
                policies, is_compliant = [], True

            samples.append(
                EvalSample(
                    job_text=job_text,
                    ground_truth_policies=policies,
                    is_compliant=is_compliant,
                    source=source,
                    row_id=i,
                )
            )

    logger.info("Loaded %d samples from %s", len(samples), path)
    return samples


def load_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    include_synthetic: bool = True,
    include_seed: bool = True,
) -> EvalDataset:
    """
    Load the eval dataset.

    Priority:
      1. User-provided data in data_dir (job_postings.csv)
      2. Synthetic violations (synthetic_violations.csv)
      3. Built-in seed dataset (30 hand-curated examples)

    The seed dataset is always available — no downloads required.
    """
    samples: list[EvalSample] = []

    # User-provided dataset
    main_path = data_dir / POSTINGS_FILE
    samples.extend(load_csv(main_path, source="kaggle"))

    # Synthetic augmentation
    if include_synthetic:
        synthetic_path = data_dir / SYNTHETIC_FILE
        samples.extend(load_csv(synthetic_path, source="synthetic"))

    # Fall back to seed dataset if nothing else loaded
    if not samples and include_seed:
        logger.info("No user data found — loading built-in seed dataset")
        samples.extend(load_csv(SEED_DATASET, source="seed"))

    dataset = EvalDataset(samples=samples)
    logger.info(dataset.summary())
    return dataset
