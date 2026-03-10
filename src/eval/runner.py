"""
Eval runner — orchestrates classification of the Indeed dataset.

Maps each job posting to the 10 Indeed policies, runs each through the
safety classifier, and compares predicted labels against ground truth.
"""

from __future__ import annotations

import asyncio
import logging
import time

from src.eval.dataset import EvalDataset, EvalSample
from src.eval.indeed_policies import INDEED_POLICIES, IndeedPolicy
from src.eval.metrics import ClassificationResult, EvalResults
from src.safety.classifier import SafetyClassifier

logger = logging.getLogger(__name__)

# Concurrency limit to avoid hammering the API
DEFAULT_CONCURRENCY = 5


async def classify_sample_against_policy(
    classifier: SafetyClassifier,
    sample: EvalSample,
    policy: IndeedPolicy,
    semaphore: asyncio.Semaphore,
) -> ClassificationResult:
    """
    Classify a single sample against a single Indeed policy.

    The sample's job_text is treated as the "model_output" (content to evaluate).
    The policy description becomes the classification context.
    We use the first mapped GA Guard category for the classifier call.
    """
    async with semaphore:
        # Build the prompt for the classifier
        prompt = (
            f"Evaluate this job posting for violations of the '{policy.name}' policy.\n"
            f"Policy: {policy.description}"
        )

        # Use the primary GA Guard category for this Indeed policy
        category = policy.ga_guard_categories[0] if policy.ga_guard_categories else "hate"

        result = await classifier.classify(
            prompt=prompt,
            model_output=sample.job_text,
            category=category,
        )

        ground_truth_unsafe = policy.id in sample.ground_truth_policies

        return ClassificationResult(
            sample_id=sample.row_id,
            policy_id=policy.id,
            predicted_unsafe=result.get("unsafe", False),
            ground_truth_unsafe=ground_truth_unsafe,
            severity=result.get("severity", 0),
            explanation=result.get("explanation", ""),
        )


async def run_eval(
    classifier: SafetyClassifier,
    dataset: EvalDataset,
    policies: list[str] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress_callback: callable | None = None,
) -> EvalResults:
    """
    Run the full eval: classify every sample against every relevant policy.

    Args:
        classifier: SafetyClassifier instance
        dataset: Loaded EvalDataset
        policies: Subset of Indeed policy IDs to evaluate (None = all 10)
        concurrency: Max concurrent API calls
        progress_callback: Optional callback(completed, total) for progress

    Returns:
        EvalResults with per-policy and overall metrics
    """
    results = EvalResults(total_samples=dataset.total)
    semaphore = asyncio.Semaphore(concurrency)

    # Determine which policies to evaluate
    policy_ids = policies or list(INDEED_POLICIES.keys())
    target_policies = [
        INDEED_POLICIES[pid] for pid in policy_ids if pid in INDEED_POLICIES
    ]

    if not target_policies:
        logger.error("No valid policies to evaluate")
        return results

    logger.info(
        "Starting eval: %d samples × %d policies = %d classifications",
        dataset.total,
        len(target_policies),
        dataset.total * len(target_policies),
    )

    total = dataset.total * len(target_policies)
    completed = 0
    start_time = time.monotonic()

    # Build all classification tasks
    tasks: list[asyncio.Task] = []
    for sample in dataset.samples:
        for policy in target_policies:
            task = asyncio.create_task(
                classify_sample_against_policy(
                    classifier, sample, policy, semaphore
                )
            )
            tasks.append(task)

    # Gather results with progress tracking
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.add_result(result)
        completed += 1

        if progress_callback:
            progress_callback(completed, total)
        elif completed % 50 == 0 or completed == total:
            elapsed = time.monotonic() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f/s) — F1: %.4f",
                completed,
                total,
                rate,
                results.overall.f1,
            )

    elapsed = time.monotonic() - start_time
    logger.info(
        "Eval complete: %d classifications in %.1fs (%.1f/s)",
        total,
        elapsed,
        total / elapsed if elapsed > 0 else 0,
    )

    return results
