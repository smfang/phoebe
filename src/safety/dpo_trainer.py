"""
DPO Trainer for Sara Safety RL.

Wraps TRL's ``DPOTrainer`` to fine-tune a model on preference pairs produced by
:mod:`src.safety.safety_rl_pipeline`.  Loads a dataset from disk (or pulls
fresh via the pipeline), tokenises it, and runs DPO training with configurable
hyperparameters.
"""

import logging
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from src.config import CONFIG

logger = logging.getLogger(__name__)


def load_or_pull(
    data_dir: str = "data/safety_rl",
    sources: list[str] | None = None,
    max_per_source: int = 10_000,
) -> DatasetDict:
    """Return an existing dataset or pull a fresh one."""
    path = Path(data_dir)
    if path.exists() and (path / "train").exists():
        logger.info("Loading cached dataset from %s", path)
        return DatasetDict.load_from_disk(str(path))

    logger.info("No cached dataset — pulling fresh data ...")
    from src.safety.safety_rl_pipeline import pull_and_format

    return pull_and_format(
        sources=sources,
        max_per_source=max_per_source,
        output_dir=data_dir,
    )


def run_dpo_training(
    model_name: str | None = None,
    data_dir: str = "data/safety_rl",
    output_dir: str = "models/safety_dpo",
    sources: list[str] | None = None,
    max_per_source: int = 10_000,
    learning_rate: float = 5e-7,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    beta: float = 0.1,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    bf16: bool = True,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 250,
) -> str:
    """
    Run DPO training on the safety preference dataset.

    Parameters
    ----------
    model_name
        HuggingFace model ID or local path.  Falls back to
        ``CONFIG.dpo_base_model`` then a sensible default.
    data_dir
        Path to the DPO dataset on disk (``train`` / ``test`` splits).
    output_dir
        Where to save the fine-tuned model + checkpoints.
    sources
        Dataset sources to include (passed through to pipeline if pulling).
    max_per_source
        Cap per dataset source.
    learning_rate, num_train_epochs, per_device_train_batch_size,
    gradient_accumulation_steps, beta, max_length, max_prompt_length,
    bf16, logging_steps, save_steps, eval_steps
        Standard DPO / training hyperparameters.

    Returns
    -------
    str
        Path to the saved model directory.
    """
    # Lazy imports — these are heavy and only needed at training time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    resolved_model = (
        model_name
        or CONFIG.dpo_base_model
        or "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    logger.info("Base model: %s", resolved_model)

    # 1. Dataset
    ds = load_or_pull(data_dir=data_dir, sources=sources, max_per_source=max_per_source)
    train_ds = ds["train"]
    eval_ds = ds["test"]
    logger.info("Train: %d examples, Eval: %d examples", len(train_ds), len(eval_ds))

    # 2. Model + tokeniser
    logger.info("Loading model and tokeniser ...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(resolved_model)

    # 3. Training config
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        bf16=bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps",
        remove_unused_columns=False,
        report_to="none",
    )

    # 4. Train
    logger.info("Starting DPO training ...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    # 5. Save
    out = Path(output_dir) / "final"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Model saved to %s", out)

    return str(out)
