from pathlib import Path
from typing import Tuple
import random, copy, uuid

from src.data.dpo_dataset import DPODataset, DPOPreferencePair
from src.data.dpo_templates import get_all_template_pairs


def build_dataset(
    augmentation_factor: int = 2,
    seed: int = 42,
    require_cot: bool = True,      # filter out pairs without thinking_trace
) -> DPODataset:
    """
    Build DPO dataset from templates with domain-swap augmentation.
    require_cot=True enforces that all training pairs have <thinking> blocks —
    this is the CoT SFT requirement from the SafeWork-inspired training design.
    """
    random.seed(seed)
    base_pairs = get_all_template_pairs()
    if require_cot:
        base_pairs = [p for p in base_pairs if p.thinking_trace]

    dataset = DPODataset(base_pairs)
    domains = ["finance", "healthcare", "dao_governance",
               "government", "insurance", "energy"]

    augmented = []
    for pair in base_pairs:
        for _ in range(augmentation_factor - 1):
            new_domain = random.choice(
                [d for d in domains if d != pair.routing_context.domain]
            )
            new_chosen   = pair.chosen.replace(
                pair.routing_context.domain, new_domain)
            new_rejected = pair.rejected.replace(
                pair.routing_context.domain, new_domain)
            new_ctx = copy.deepcopy(pair.routing_context)
            new_ctx.domain = new_domain
            augmented.append(DPOPreferencePair(
                pair_id=str(uuid.uuid4()),
                chosen=new_chosen,
                rejected=new_rejected,
                atlas_tactic_label=pair.atlas_tactic_label,
                routing_context=new_ctx,
                severity=pair.severity,
                dao_category=pair.dao_category,
                thinking_trace=pair.thinking_trace,
                source="augmented",
                tags=pair.tags + [f"augmented_from:{pair.pair_id}"],
            ))
    for p in augmented:
        dataset.add(p)
    return dataset


def split_dataset(
    dataset: DPODataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DPODataset, DPODataset, DPODataset]:
    random.seed(seed)
    pairs = dataset.pairs.copy()
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)
    return (
        DPODataset(pairs[:train_end]),
        DPODataset(pairs[train_end:val_end]),
        DPODataset(pairs[val_end:]),
    )


def save_splits(
    output_dir: str = "data/dpo",
    augmentation_factor: int = 2,
) -> dict:
    dataset = build_dataset(augmentation_factor=augmentation_factor)
    train, val, test = split_dataset(dataset)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, split in [("train", train), ("val", val),
                        ("test", test), ("full", dataset)]:
        path = f"{output_dir}/{name}.json"
        split.save(path)
        paths[name] = path
        print(f"Saved {name}: {len(split)} pairs → {path}")
    print(f"\nDataset stats:\n{dataset.get_stats()}")
    return paths


if __name__ == "__main__":
    save_splits()
