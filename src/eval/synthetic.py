"""
Synthetic violation generator for augmenting the Indeed eval dataset.

The Kaggle dataset is >92% compliant, so we need synthetic violations
to get meaningful precision/recall numbers. Two approaches:

1. Generate from templates (this module) — fast, deterministic
2. Use Phoebe's red team mode (python main.py redteam --auto) — LLM-generated

GA supplemented their dataset with ~2K synthetic violations using their
adversarial RL pipeline. We can do a lighter version with templates.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

from src.eval.indeed_policies import INDEED_POLICIES, IndeedPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Templates for synthetic violations per policy
# ---------------------------------------------------------------------------

VIOLATION_TEMPLATES: dict[str, list[str]] = {
    "discrimination": [
        "We're looking for a {adj} {gender} to join our team. Age {age_range} preferred.",
        "This role requires someone of {religion} faith. {nationality} speakers only.",
        "No applicants over {max_age}. We need {adj} team members who fit our culture.",
        "{gender} only need apply. Must be physically {appearance}.",
        "Seeking a {adj} {age_range}-year-old for our fast-paced environment.",
    ],
    "misleading_info": [
        "EARN ${income}/WEEK FROM HOME! No experience needed! Be your own boss!",
        "Guaranteed ${income}/month income! Just {hours} hours/week!",
        "{title} position — unlimited earning potential! Start immediately!",
        "Work from home and make ${income}/day! 100% legitimate opportunity!",
        "Secret shopper needed — earn ${income}/assignment, unlimited assignments!",
    ],
    "illegal_activities": [
        "Package forwarding agent needed. Receive shipments and reship internationally. ${income}/week.",
        "Fund transfer specialist — move money between accounts for our global clients. No banking license required.",
        "Process payments for our online {business}. No questions asked. Cash weekly.",
        "Help us {activity}. High pay, no background check needed.",
        "Cryptocurrency mixer operator needed. Knowledge of privacy tools preferred.",
    ],
    "pii_harvesting": [
        "To apply, email your SSN, DOB, and bank routing number to {email}.",
        "Application requires: photo of driver's license, credit card number for background check fee (${fee}).",
        "Please provide your mother's maiden name and SSN for verification. Fee: ${fee}.",
        "Submit your passport copy, bank details, and {additional_doc} to start immediately.",
        "We need your Social Security Number and bank login for direct deposit setup BEFORE hiring.",
    ],
    "sexual_content": [
        "Attractive {gender} needed for private companionship. Discretion guaranteed. ${income}/hr.",
        "Adult entertainment {role} — must be comfortable with explicit content. High pay.",
        "Looking for {adj} models for private events. No experience needed, just {appearance}.",
        "Escort service seeking new {gender} companions. Premium clients only.",
        "Webcam {role} wanted — earn ${income}/day from home. Must be attractive.",
    ],
    "violence_threats": [
        "Debt recovery specialist needed. Must be {adj} and willing to use {method} to collect.",
        "Security enforcer — physical intimidation required. No formal training needed.",
        "Need someone tough for {activity}. Must not be squeamish about confrontation.",
        "Collections agent — visit debtors at home, use {method} to ensure payment.",
        "Bouncer needed — must handle {adj} situations. Previous combat experience preferred.",
    ],
    "scam_postings": [
        "Invest ${fee} in your starter kit and earn ${income}/month passive income!",
        "Recruit {count} people below you and earn from their sales forever!",
        "Send ${fee} processing fee to secure your position. Start earning ${income}/week!",
        "Build your downline of {count}+ members for lifetime passive income!",
        "One-time investment of ${fee} required. Returns of {return_pct}% guaranteed monthly!",
    ],
    "hate_speech": [
        "No {group} need apply. This is a {adj} workplace.",
        "{group} are not welcome. We maintain traditional values only.",
        "Seeking staff who share our {adj} views. {group} will not fit in.",
        "This position is not suitable for {group}. Apply only if you're {adj}.",
        "We do not hire {group}. Our clients prefer {adj} service providers.",
    ],
    "privacy_violations": [
        "All employees must provide social media passwords for monitoring during employment.",
        "Company will install tracking software on your personal {device}. Mandatory.",
        "Must share personal email login and all {platform} credentials before start date.",
        "GPS tracking on personal vehicle required 24/7 during employment.",
        "Submit your personal {platform} account credentials for our review process.",
    ],
    "child_safety": [
        "Hiring ages {min_age}+, no work permit needed. Night shifts {hours}.",
        "Teen workers wanted for {hazard_work}. No parental consent needed.",
        "Young workers {min_age}+ for {hazard_work}. Overtime available, flexible rules.",
        "Students age {min_age}+ for {hours} warehouse shifts. Heavy machinery training provided.",
        "Minors welcome! No age verification. Flexible hours including {hours}.",
    ],
}

# Fill-in values for templates
FILL_VALUES: dict[str, list[str]] = {
    "adj": ["young", "energetic", "attractive", "fit", "healthy", "strong", "white"],
    "gender": ["female", "male", "women", "men", "girls", "ladies"],
    "age_range": ["18-25", "21-30", "under 35", "25-40"],
    "max_age": ["40", "35", "45", "30"],
    "religion": ["Christian", "Catholic", "Muslim", "Hindu"],
    "nationality": ["American", "native English", "white American"],
    "appearance": ["attractive", "fit", "slim", "good-looking"],
    "income": ["5000", "3000", "8000", "10000", "2000"],
    "hours": ["2-3", "4-5", "10", "midnight-6am", "11pm-7am"],
    "title": ["Executive Assistant", "Marketing Manager", "Sales Director"],
    "business": ["pharmacy", "lending company", "investment firm"],
    "activity": ["distribute products", "collect debts", "recruit members"],
    "email": ["hr@example.com", "apply@fastcash.biz"],
    "fee": ["200", "500", "99", "350", "1000"],
    "additional_doc": ["utility bill", "birth certificate"],
    "role": ["performer", "model", "host", "presenter"],
    "method": ["intimidation", "any means necessary", "physical persuasion", "force"],
    "count": ["5", "10", "3", "7"],
    "return_pct": ["50", "100", "200", "30"],
    "group": ["foreigners", "immigrants", "outsiders", "those people"],
    "device": ["phone", "laptop", "tablet", "computer"],
    "platform": ["Facebook", "Instagram", "email", "social media"],
    "min_age": ["13", "14", "12", "15"],
    "hazard_work": ["warehouse work with forklifts", "construction site cleanup", "chemical handling", "heavy machinery operation"],
}


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a template with random values."""
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, rng.choice(values), 1)
    return result


def generate_synthetic_violations(
    n_per_policy: int = 200,
    seed: int = 42,
    output_path: Path | None = None,
) -> list[dict[str, str]]:
    """
    Generate synthetic violation examples for each Indeed policy.

    Args:
        n_per_policy: Number of violations to generate per policy
        seed: Random seed
        output_path: If provided, write to CSV at this path

    Returns:
        List of dicts with job_text, ground_truth, source columns
    """
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for policy_id, templates in VIOLATION_TEMPLATES.items():
        for i in range(n_per_policy):
            template = rng.choice(templates)
            job_text = _fill_template(template, rng)
            rows.append({
                "job_text": job_text,
                "ground_truth": policy_id,
                "source": "synthetic",
            })

    rng.shuffle(rows)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["job_text", "ground_truth", "source"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(
            "Generated %d synthetic violations → %s",
            len(rows),
            output_path,
        )

    return rows
