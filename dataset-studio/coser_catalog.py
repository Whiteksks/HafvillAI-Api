"""Select source books by title, author and screened PG edition metadata.

This is book-level selection. It does not assert that CoSER's rewritten
dialogue is verbatim text of a particular Gutenberg translation/edition.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from coser_download import DEFAULT_ROOT, REVISION
from data_utils import atomic_json, normalize, sha256
from gutenberg import author_forms, cached_edition, read_catalog, title_variants
from gutenberg import EditionStore

POLICY = "coser-pg-title-author-edition-v1"


def match_author(candidates, profiles):
    text = f" {normalize(' '.join(profiles.values()))} "
    evidence = {}
    for candidate in candidates:
        for form in sorted(author_forms(candidate["author"])):
            if f" {form} " in text:
                evidence[candidate["author"]] = {"matched_name": form}
                break
    if len(evidence) != 1:
        return [], evidence
    author = next(iter(evidence))
    return [c for c in candidates if c["author"] == author], evidence


def build_catalog(root=DEFAULT_ROOT, refresh=False):
    root = Path(root)
    data_dir = root.parent
    catalog_path = data_dir / "pg_catalog.csv"
    catalog = read_catalog(catalog_path)
    download = json.loads((root / "download_manifest.json").read_text())
    if download["revision"] != REVISION:
        raise ValueError("CoSER revision mismatch")
    store = EditionStore(data_dir / "gutenberg/editions")
    entries, counts = [], Counter()
    for name, source in sorted(download["files"].items()):
        if not name.startswith("full/"):
            continue
        path = root / "raw" / name
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Changed CoSER source: {name}")
        title = path.stem
        candidates = {}
        for variant in title_variants(title):
            candidates.update(catalog.get(variant, {}))
        if not candidates:
            entries.append({"book": title, "file": name, "status": "no_title_match"})
            counts["no_title_match"] += 1
            continue
        data = json.loads(path.read_text())
        profiles = {character: value.get("profile", "") for character, value in data.get("character_datasets", {}).items()
                    if isinstance(value, dict) and isinstance(value.get("profile", ""), str)}
        selected, authors = match_author(list(candidates.values()), profiles)
        editions, unavailable = [], []
        for candidate in sorted(selected, key=lambda c: int(c["id"])):
            try:
                edition = store.load(candidate["id"]) if refresh else cached_edition(store, candidate["id"])
                if edition is None:
                    unavailable.append({"id": candidate["id"], "reason": "Missing/stale cached edition"})
                    continue
                edition.pop("body", None)
                if edition["status"] == "public_domain_us":
                    editions.append({"catalog": candidate, **edition})
                else:
                    unavailable.append({"id": candidate["id"], "reason": edition["reason"]})
            except (OSError, ValueError) as exc:
                unavailable.append({"id": candidate["id"], "reason": f"{type(exc).__name__}: {exc}"})
        status = "selected" if editions else "author_ambiguous_or_missing" if not selected else "edition_unconfirmed"
        counts[status] += 1
        entries.append({"book": title, "file": name, "status": status, "source_sha256": source["sha256"],
                        "author_evidence": authors, "editions": editions, "excluded_editions": unavailable})
        if selected:
            print(f"Catalog: {title}: {status} ({len(editions)} editions)", flush=True)
    result = {"policy": POLICY, "revision": REVISION, "pg_catalog_sha256": sha256(catalog_path),
              "download_manifest_sha256": sha256(root / "download_manifest.json"),
              "selection": "Exact normalized title variants + explicit author in CoSER profiles + screened PG edition.",
              "scope": "Book-level Gutenberg selection, not verbatim dialogue or exact CoSER translation clearance.",
              "counts": dict(counts), "books": entries}
    atomic_json(root / "book_catalog.json", result)
    print(json.dumps(result["counts"], ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--refresh-editions", action="store_true", help="Fetch missing or stale PG evidence")
    args = parser.parse_args()
    build_catalog(args.root, args.refresh_editions)
