"""Minimal Gray and ClearML training example."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from gray import load_config
from gray.experiment import Experiment, artifact_dir, experiment_report
from gray.runtime import seed_everything
from gray.utils import GrayLogger, write_json


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "demo.yaml"


def train(config: dict[str, Any], logger: GrayLogger) -> tuple[Path, dict[str, Any]]:
    """Run a small replaceable training loop and save its model checkpoint.

    Args:
        config: Resolved Gray experiment configuration.
        logger: Logger connected to the terminal, local file, and ClearML.

    Returns:
        The saved checkpoint path and final evaluation metrics.
    """
    seed_everything(config["train"]["seed"])
    epochs = config["train"]["epochs"]
    validation_auc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = 1.0 / epoch
        validation_auc = 0.80 + epoch * 0.03
        logger.info("epoch=%s/%s", epoch, epochs)
        logger.metric("train_loss", train_loss, iteration=epoch)
        logger.metric("validation_auc", validation_auc, iteration=epoch)

    checkpoint = artifact_dir(config, "train", create=True) / "model.ckpt"
    write_json(
        checkpoint,
        {
            "model_version": config["model"]["model_version"],
            "validation_auc": validation_auc,
        },
    )
    evaluation = {"status": "passed", "validation_auc": validation_auc}
    logger.success("training completed: %s", checkpoint)
    return checkpoint, evaluation


def main(argv: list[str] | None = None) -> None:
    """Load one Hydra config and execute a ClearML-managed training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    config, _ = load_config(args.config, args.override)
    experiment_report(config)

    with Experiment(config) as experiment:
        logger = experiment.get_logger("train")
        logger.info("experiment_id=%s", config["experiment_id"])
        logger.info("clearml_run_id=%s", experiment.run_id)
        checkpoint, evaluation = train(config, logger)
        experiment.complete(checkpoint, evaluation=evaluation)


if __name__ == "__main__":
    main()
