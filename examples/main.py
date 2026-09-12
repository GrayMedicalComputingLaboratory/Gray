"""Minimal Gray and ClearML training example."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from gray import load_config
from gray.experiment import artifact_dir, init_task
from gray.runtime import seed_everything
from gray.utils.io import write_json


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "demo.yaml"


def train(config: dict[str, Any], task: Any) -> Path:
    """Run a small replaceable training loop and save its model checkpoint.

    Args:
        config: Resolved Gray experiment configuration.
        task: Native ClearML Task for logging, metrics, and artifacts.

    Returns:
        The saved checkpoint path.
    """
    seed_everything(config["train"]["seed"])
    logger = task.get_logger()
    epochs = config["train"]["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("train.epochs must be a positive integer")
    validation_auc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = 1.0 / epoch
        validation_auc = min(1.0, 0.80 + epoch * 0.03)
        logger.report_text(f"epoch={epoch}/{epochs}")
        logger.report_scalar(title="Metrics", series="train_loss", value=train_loss, iteration=epoch)
        logger.report_scalar(title="Metrics", series="validation_auc", value=validation_auc, iteration=epoch)

    checkpoint = artifact_dir(config, "train", create=True) / "model.json"
    write_json(
        checkpoint,
        {
            "model_version": config["model"]["model_version"],
            "validation_auc": validation_auc,
        },
    )
    task.connect(
        {"status": "passed", "validation_auc": validation_auc},
        name="Evaluation",
        ignore_remote_overrides=True,
    )
    logger.report_single_value(name="validation_auc", value=validation_auc)
    task.update_output_model(
        model_path=str(checkpoint),
        model_name=config["model"]["model_version"],
        auto_delete_file=False,
    )
    logger.report_text(f"training completed: {checkpoint}")
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    """Load one Hydra config and execute a ClearML-managed training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.override)
    train(config, init_task(config))


if __name__ == "__main__":
    main()
