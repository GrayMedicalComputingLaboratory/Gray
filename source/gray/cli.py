"""A small CLI that demonstrates the shared CV lifecycle."""
from __future__ import annotations

import argparse
from pathlib import Path

from gray.core.config import artifact_dir, load_config, resolve_path
from gray.core.runtime import resolve_device
from gray.evaluators import ClassificationEvaluator
from gray.trainers import ClassificationTrainer
from gray.utils.runtime import get_logger, write_json


def _paths(config: dict) -> tuple[Path, Path, Path]:
    output = artifact_dir(config, "models")
    train = resolve_path(config, config["dataset"]["train_manifest"])
    valid = resolve_path(config, config["dataset"].get("valid_manifest", config["dataset"]["train_manifest"]))
    return output, train, valid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gray")
    parser.add_argument("command", choices=("train", "validate", "analyze"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config, _ = load_config(args.config); output, train, valid = _paths(config)
    device = resolve_device(config["runtime"].get("device", "cpu"))
    logger = get_logger("gray", artifact_dir(config, "logs", create=True))
    checkpoint = output / f"{config['experiment_id']}_baseline.npz"
    if args.command == "train":
        logger.info("experiment=%s device=%s stage=train start", config["experiment_id"], device)
        result = ClassificationTrainer(train, config).fit()
        logger.info("single-run baseline: complete OOF summary is not applicable")
        logger.info("stage=train complete %s", result)
        return
    if not checkpoint.exists(): raise FileNotFoundError(f"run train first: {checkpoint}")
    if args.command == "validate": print(ClassificationEvaluator(valid, checkpoint, artifact_dir(config, "validation", create=True)).evaluate()); return
    if args.command == "analyze":
        analysis_dir = artifact_dir(config, "analysis", create=True)
        metrics = ClassificationEvaluator(valid, checkpoint, analysis_dir).evaluate()
        write_json(analysis_dir / "report.json", {"task": "classification", "metrics": metrics}); print(metrics); return
if __name__ == "__main__": main()
