"""File persistence, hashing, and stable text identifiers for CoSER preparation."""

from datetime import datetime, timezone
import hashlib
import json
import os
import re
import unicodedata

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c if c.isalnum() else " " for c in text
                            if not unicodedata.combining(c)).split())


def source_book_key(title):
    # Keep stable source-book identifiers used by existing CoSER exports.
    replacements = {"\u200b": "", "’": "'", "‘": "'", "“": '"', "”": '"',
                    "—": "-", "–": "-", "…": "...", "!": "", "?": "", ";": ",", " ": "_"}
    for old, new in replacements.items():
        title = title.replace(old, new)
    return re.sub(r"_+", "_", title).strip("_")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()
