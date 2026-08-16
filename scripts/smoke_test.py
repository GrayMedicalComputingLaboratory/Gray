"""Run the complete example lifecycle from a checkout without installation."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
config = root / "source" / "configs" / "example_classification" / "example_classification.yaml"
subprocess.run([sys.executable, str(root / "scripts" / "create_example_images.py")], check=True)
for command in ("train", "validate", "analyze"):
    subprocess.run([sys.executable, "-m", "graycv.cli", command, "--config", str(config)], check=True, cwd=root)
