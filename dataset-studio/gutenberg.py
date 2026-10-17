"""Book catalog matching and cached Gutenberg edition screening for CoSER."""

from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from html.parser import HTMLParser
import json
import re
import urllib.request

from data_utils import atomic_json, normalize, sha256

CACHE_DAYS = 30

def title_variants(title):
    # No fuzzy matches, substring matches, dropped articles or invented aliases.
    title = re.sub(r"\s*\([^)]*#\d[^)]*\)\s*$", "", title).strip()
    variants = {title, title.splitlines()[0]}
    for value in tuple(variants):
        variants.add(value.split(":", 1)[0])
        variants.add(re.split(r";\s*or\b", value, maxsplit=1, flags=re.I)[0])
    return {normalize(value) for value in variants if normalize(value)}


@lru_cache(maxsize=4096)
def author_forms(author):
    """Catalog name → explicit full-name forms, including expanded initials."""
    if re.search(r"\[(?:Editor|Translator|Illustrator|Contributor)", author, re.I):
        return set()
    parts = author.split(",")
    surname = parts[0].strip()
    if normalize(surname) in {"anonymous", "various", "unknown", ""}:
        return set()
    if len(parts) < 2 or re.search(r"\d", parts[1]):
        return {normalize(surname)}  # e.g. Homer or Stendhal
    given = parts[1].strip()
    abbreviated = re.sub(r"\([^)]*\)", "", given).strip()
    expanded = re.search(r"\(([^)]+)\)", given)
    names = {abbreviated}
    if expanded:
        names.add(expanded.group(1))
    return {normalize(f"{first} {surname}") for first in names if first}


def read_catalog(path):
    by_title = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        required = {"Text#", "Type", "Title", "Language", "Authors"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError("The Gutenberg CSV is missing required columns.")
        for row in rows:
            languages = [lang.strip() for lang in row["Language"].split(";")]
            if row["Type"] != "Text" or "en" not in languages:
                continue
            primary = row["Authors"].split(";")[0].strip()
            entry = {"id": row["Text#"], "title": row["Title"], "author": primary,
                     "authors": row["Authors"], "language": row["Language"]}
            for title in title_variants(row["Title"]):
                by_title[title][row["Text#"]] = entry
    return by_title


def utcnow():
    return datetime.now(timezone.utc)


def fresh(timestamp):
    try:
        checked = datetime.fromisoformat(timestamp)
        age = utcnow() - checked
        return timedelta(0) <= age <= timedelta(days=CACHE_DAYS)
    except (TypeError, ValueError):
        return False


class CardParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rights = []
        self.capturing = False

    def handle_starttag(self, tag, attrs):
        if tag == "td" and dict(attrs).get("property") == "dcterms:rights":
            self.capturing = True

    def handle_endtag(self, tag):
        if tag == "td":
            self.capturing = False

    def handle_data(self, data):
        if self.capturing:
            self.rights.append(data)


def inspect_edition(card, text, identifier):
    parser = CardParser()
    parser.feed(card)
    rights = " ".join(" ".join(parser.rights).split())
    # The license/footer must stay outside the text used for comparison.
    start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    end = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    declared_id = re.search(r"\[(?:EBook|eBook)\s*#\s*(\d+)\]", text, re.I)
    reason = None
    if not declared_id or declared_id.group(1) != identifier:
        reason = "The Gutenberg file's declared ID is missing or different."
    elif not start or not end or end.start() <= start.end():
        reason = "Recognized Gutenberg start/end markers are missing."
    elif rights != "Public domain in the USA.":
        reason = "The Gutenberg card does not declare public domain in the USA."
    # Do not confuse the standard PG footer (which discusses all works) with
    # a work-specific copyright declaration in this book's header or body.
    pre_footer = text[:end.start()] if end else text
    special = re.search(r"this is a copyrighted|translation\s+copyright|"
                        r"creative\s+commons|all rights reserved|non[ -]?commercial|personal use only|"
                        r"copyright[\s\S]{0,120}?(?:19[3-9]\d|20\d\d)|"
                        r"copyright\s*(?:©|\(c\))?\s*(?:by|:)", pre_footer, re.I)
    if special:
        reason = "A work-specific rights notice needs review; public-domain metadata alone is insufficient."
    return {"copyright": rights or "Unknown", "status": "review" if reason else "public_domain_us",
            "reason": reason or "PG card and ebook notices passed US public-domain screening.",
            "notice_excerpt": pre_footer[max(0, special.start() - 80):special.end() + 160] if special else None,
            "body": text[start.end():end.start()] if start and end else ""}


def fetch_bytes(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "Hafvillai-Dataset-Studio/1.0 (source verification)"})
    with urllib.request.urlopen(request, timeout=20) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Gutenberg response exceeds the verification size limit.")
    return content


class EditionStore:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def load(self, identifier):
        if not re.fullmatch(r"\d+", identifier):
            raise ValueError("Invalid Gutenberg identifier.")
        path = self.root / f"{identifier}.json"
        cached = json.loads(path.read_text()) if path.exists() else None
        card_path, text_path = self.root / f"{identifier}.html", self.root / f"{identifier}.txt"
        valid = cached and fresh(cached.get("checked_at")) and card_path.exists() and text_path.exists()
        if valid:
            valid = sha256(card_path) == cached.get("card_sha256") and sha256(text_path) == cached.get("text_sha256")
        if not valid:
            card = fetch_bytes(f"https://www.gutenberg.org/ebooks/{identifier}", 1_000_000)
            text = fetch_bytes(f"https://mirror.cs.odu.edu/gutenberg-epub/{identifier}/pg{identifier}.txt", 16_000_000)
            for target, content in ((card_path, card), (text_path, text)):
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(content)
                temporary.replace(target)
            cached = {"id": identifier, "checked_at": utcnow().isoformat(),
                      "card_url": f"https://www.gutenberg.org/ebooks/{identifier}",
                      "text_url": f"https://mirror.cs.odu.edu/gutenberg-epub/{identifier}/pg{identifier}.txt",
                      "card_sha256": sha256(card_path), "text_sha256": sha256(text_path)}
            atomic_json(path, cached)
        assessment = inspect_edition(card_path.read_text(encoding="utf-8"),
                                     text_path.read_text(encoding="utf-8-sig"), identifier)
        return {**cached, **assessment}


def cached_edition(store, identifier):
    """Preparation is offline: never refresh a stale edition implicitly."""
    if not identifier.isdigit():
        raise ValueError("Invalid PG identifier")
    metadata = store.root / f"{identifier}.json"
    if not metadata.exists():
        return None
    cached = json.loads(metadata.read_text())
    if not fresh(cached.get("checked_at")):
        return None
    for suffix, field in [("html", "card_sha256"), ("txt", "text_sha256")]:
        path = store.root / f"{identifier}.{suffix}"
        if not path.exists() or sha256(path) != cached.get(field):
            return None
    edition = store.load(identifier)
    return edition if edition["status"] == "public_domain_us" else None
