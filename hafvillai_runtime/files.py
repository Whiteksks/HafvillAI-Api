"""Checksums shared by model loading and artifact preparation."""

import hashlib
from pathlib import Path


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
