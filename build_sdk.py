"""Build the SDK installation file: python build_sdk.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Create an installation file in dist/sdk/ using the current Python."""
    if importlib.util.find_spec("build") is None:
        print(
            "First install the packaging tool: python -m pip install build",
            file=sys.stderr,
        )
        return 1
    root = Path(__file__).resolve().parent
    destination = root / "dist" / "sdk"
    print("Building the SDK installation file...", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            str(root / "source" / "gray"),
            "--outdir",
            str(destination),
        ],
        check=False,
    )
    if result.returncode == 0:
        print(f"Installation file saved in: {destination}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
