#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/create_example_images.py
python -m graycv.cli "${1:-train}" --config source/configs/example_classification/example_classification.yaml
