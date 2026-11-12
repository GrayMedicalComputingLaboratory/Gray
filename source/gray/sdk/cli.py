"""Standalone inference CLI; never persists model inputs or logs tensors."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gray.utils.io import write_json

from .exceptions import GraySDKError
from .image import load_image
from .runtime import load_model
from .types import ImageInput


def _count(value: str) -> int:
    """Parse a nonnegative integer CLI option."""
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def main(argv: list[str] | None = None) -> int:
    """Run infer/info/speed or legacy commands; return 0/1/2 exit codes.

    inspect opens the engine and checks the model contract. infer writes only
    outputs.npz and metadata.json to a newly created output directory.
    """
    parser = argparse.ArgumentParser(
        prog="gray-sdk",
        description="Run a model, view its details, or measure its speed.",
        epilog="Start here: gray-sdk infer --model MODEL_FOLDER --input IMAGE.png",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("infer", "Run inference and save the result"),
        ("info", "Show model details"),
        ("speed", "Measure prediction speed (uses sensible defaults)"),
    ):
        sub = commands.add_parser(command, help=help_text, description=help_text)
        model_args = (
            sub.add_mutually_exclusive_group(required=True)
            if command == "infer"
            else sub
        )
        model_args.add_argument(
            "--model",
            type=Path,
            required=command != "infer",
            help="Model folder or model settings YAML file",
        )
        if command == "infer":
            model_args.add_argument(
                "--manifest",
                type=Path,
                help="Model settings file (compatibility option)",
            )
        sub.add_argument(
            "--device", default="cpu", help="cpu (default), cuda, or cuda:0"
        )
        if command != "info":
            sub.add_argument(
                "--input", type=Path, required=True, help="Input image or .npy array"
            )
            sub.add_argument(
                "--dim",
                help="Array dimensions, for example HW or DHW; defaults to model settings",
            )
        if command == "infer":
            sub.add_argument(
                "--output",
                type=Path,
                help="Where to save results; default: a new folder inside outputs/",
            )
        elif command == "speed":
            sub.add_argument(
                "--runs",
                type=_count,
                default=100,
                help="Number of measured runs (default: 100)",
            )
    inspect = commands.add_parser("inspect", help="Validate manifest and ONNX session")
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument("--device", default="cpu")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument(
        "--input", type=Path, required=True, help="Image file or numeric .npy array"
    )
    benchmark.add_argument("--dim", help="Input dimensions, for example HW or DHW")
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--warmup", type=_count, default=10)
    benchmark.add_argument("--iterations", type=_count, default=100)
    args = parser.parse_args(argv)
    simple = args.command in ("info", "speed") or (
        args.command == "infer" and args.model is not None
    )
    if args.command == "infer" and not simple and args.output is None:
        parser.error("--output is required with --manifest")
    if simple:
        args.manifest = args.model
        args.command = {"infer": "infer", "info": "inspect", "speed": "benchmark"}[
            args.command
        ]
        if args.command == "benchmark":
            args.iterations = args.runs
            args.warmup = 10
        elif args.command == "infer" and args.output is None:
            args.output = Path("outputs") / datetime.now().strftime(
                "sdk-%Y%m%d-%H%M%S-%f"
            )
    if args.command == "benchmark" and args.iterations == 0:
        parser.error(
            "--runs must be positive" if simple else "--iterations must be positive"
        )
    try:
        with load_model(args.manifest, device=args.device) as runtime:
            report: dict[str, Any]
            if args.command == "inspect":
                report = {
                    "model": runtime.manifest.model,
                    "sha256": runtime.model_sha256,
                    "input": runtime.manifest.input,
                    "outputs": runtime.manifest.outputs,
                    "device": runtime.device,
                }
            else:
                source = args.input.expanduser().resolve(strict=True)
                if not source.is_file():
                    raise ValueError("--input must be an existing image or .npy file")
                if source.suffix.lower() == ".npy":
                    data = np.load(source, allow_pickle=False)
                    image = ImageInput(
                        data, args.dim or runtime.manifest.input["dim"]
                    )
                else:
                    image = load_image(source, dim=args.dim)
                if args.command == "infer":
                    result = runtime.predict(image)
                    destination = args.output.expanduser().resolve()
                    destination.mkdir(parents=True, exist_ok=False)
                    np.savez(destination / "outputs.npz", **result.outputs)
                    report = {
                        "task_type": result.task_type,
                        "latency_ms": result.latency_ms,
                        "metadata": result.metadata,
                    }
                    write_json(destination / "metadata.json", report)
                else:
                    for _ in range(args.warmup):
                        runtime.predict(image)
                    times = np.array(
                        [
                            runtime.predict(image).latency_ms
                            for _ in range(args.iterations)
                        ]
                    )
                    report = {
                        "iterations": args.iterations,
                        "warmup": args.warmup,
                        "mean_ms": float(times.mean()),
                        "p50_ms": float(np.percentile(times, 50)),
                        "p95_ms": float(np.percentile(times, 95)),
                        "device": runtime.device,
                        "timing": "validation + preprocessing + session.run",
                    }
            if not simple:
                print(json.dumps(report, indent=2))
            elif args.command == "inspect":
                print(f"Model: {runtime.manifest.model['name']}")
                print(f"Version: {runtime.manifest.model['version']}")
                print(f"Task: {runtime.manifest.model['task_type']}")
                print(
                    f"Input: {runtime.manifest.input['dim']} {runtime.manifest.input['shape']} {runtime.manifest.input['dtype']}"
                )
                print(f"Device: {runtime.device}")
            elif args.command == "infer":
                print(f"Result saved to: {destination}")
                print(f"Time: {result.latency_ms:.3f} ms")
            else:
                print(f"Runs: {args.iterations}")
                print(f"Average time: {report['mean_ms']:.3f} ms")
                print(f"95% of runs finished within: {report['p95_ms']:.3f} ms")
        return 0
    except (GraySDKError, OSError, ValueError) as error:
        print(f"gray-sdk: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
