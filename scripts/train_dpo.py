"""
Sheila DPO Judge Training Script

Target model: Qwen/Qwen2.5-7B-Instruct (Sheila's judge base model)
Estimated cost: $10–30/run via Replicate or Modal

Note: DeepSeek deepseek-chat is deprecated 2026-07-24 — do not use as base.
Use Qwen2.5-7B-Instruct as the default base model.

Usage:
  python scripts/train_dpo.py \\
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
    --train_data data/dpo/train.json \\
    --val_data data/dpo/val.json \\
    --output_dir models/sheila-judge-dpo

LoRA config:
  r=16, lora_alpha=32, target_modules=["q_proj","v_proj"]
  lora_dropout=0.05, task_type=CAUSAL_LM

DPO config:
  beta=0.1 (default), max_length=1024, max_prompt_length=512
  gradient_accumulation_steps=4, report_to=none (default)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Sheila judge model with DPO on Sara safety preference dataset"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/dpo/train.json",
        help="Path to training data JSON",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/dpo/val.json",
        help="Path to validation data JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/sheila-judge-dpo",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--dpo_beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (default: 0.1)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max sequence length (default: 1024)",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Max prompt length (default: 512)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Reporting backend: none, wandb, tensorboard (default: none)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Per-device training batch size (default: 1)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    return parser.parse_args()


def load_dpo_data(path: str):
    """Load DPO dataset from JSON file and convert to HuggingFace format."""
    with open(path) as f:
        records = json.load(f)

    from src.data.dpo_dataset import (
        DPODataset, ATLASTactic, RiskTier, RoutingContext, DPOPreferencePair
    )

    dataset = DPODataset.load(path)
    training_list = dataset.to_training_list()
    logger.info("Loaded %d pairs from %s", len(training_list), path)
    return training_list


def build_lora_config(args):
    """Build LoRA configuration."""
    from peft import LoraConfig, TaskType
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def train(args):
    """Main training function."""
    logger.info("Starting Sheila DPO training")
    logger.info("Base model: %s", args.model_name_or_path)
    logger.info("Output: %s", args.output_dir)

    # Check dependencies
    try:
        import torch
        import transformers
        import trl
        import peft
        import datasets
    except ImportError as e:
        logger.error(
            "Missing dependency: %s\n"
            "Install with: pip install trl>=0.7.0 peft>=0.6.0 transformers>=4.36.0 "
            "datasets>=2.14.0 torch",
            e,
        )
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer, DPOConfig
    from peft import get_peft_model
    import datasets as hf_datasets

    # Load data
    train_data = load_dpo_data(args.train_data)
    val_data = load_dpo_data(args.val_data)

    train_ds = hf_datasets.Dataset.from_list(train_data)
    val_ds = hf_datasets.Dataset.from_list(val_data)

    # Load model and tokenizer
    logger.info("Loading tokenizer from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s", args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Apply LoRA
    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # DPO training config
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        report_to=args.report_to,
        beta=args.dpo_beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # use implicit reference (same model with frozen LoRA base)
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model saved to %s", args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
