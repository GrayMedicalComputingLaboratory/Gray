"""Streaming file hashing utilities."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: str | Path) -> str:
    """Calculate the SHA-256 checksum of a file without loading it all at once.

    Args:
        path: File to hash.

    Returns:
        The lowercase hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        IsADirectoryError: If ``path`` is a directory.
        OSError: If the file cannot be read.
    """
    source = Path(path).expanduser()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
