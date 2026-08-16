"""Create tiny deterministic images for the framework smoke-test example."""
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[1] / "examples" / "classification" / "images"
root.mkdir(parents=True, exist_ok=True)
for name, color in {"red_1": (220, 30, 30), "red_2": (180, 20, 20), "blue_1": (30, 60, 220), "blue_2": (20, 40, 180)}.items():
    Image.new("RGB", (64, 64), color).save(root / f"{name}.png")
