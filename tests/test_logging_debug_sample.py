"""Tests for Gray diagnostic image reporting."""

from __future__ import annotations

from pathlib import Path

from gray.utils.logging import get_logger


def test_debug_sample_reports_local_image_and_flushes(tmp_path: Path) -> None:
    """Forward a local diagnostic image through the configured tracker."""
    sample = tmp_path / "sample.png"
    sample.write_bytes(b"png")

    class Tracker:
        def __init__(self) -> None:
            self.images: list[dict[str, object]] = []
            self.flushes: list[bool] = []

        def report_text(self, message: str, **options: object) -> None:
            del message, options

        def report_scalar(self, **options: object) -> None:
            del options

        def report_image(self, **options: object) -> None:
            self.images.append(options)

        def flush(self, *, wait: bool) -> None:
            self.flushes.append(wait)

    tracker = Tracker()
    logger = get_logger("debug-sample", tmp_path, tracker=tracker)
    logger.debug_sample("Validation", "case_001", sample, iteration=2)
    logger.close()

    assert tracker.images == [
        {
            "title": "Validation",
            "series": "case_001",
            "iteration": 2,
            "local_path": str(sample.resolve()),
            "max_image_history": 5,
        }
    ]
    assert tracker.flushes == [False]
