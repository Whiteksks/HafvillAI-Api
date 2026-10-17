"""Gutenberg screening and offline evidence checks independent of dataset exports."""

from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_utils import atomic_json, sha256
from gutenberg import EditionStore, cached_edition, inspect_edition, read_catalog, utcnow

CARD = '<td property="dcterms:rights">Public domain in the USA.</td>'
TEXT = (
    "[EBook #123]\n*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
    "Two friends met in the library and exchanged letters.\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
)


class GutenbergTests(unittest.TestCase):
    def test_screening_rejects_wrong_ids_missing_markers_and_conflicting_notices(self):
        self.assertEqual(inspect_edition(CARD, TEXT, "123")["status"], "public_domain_us")
        for card, text, identifier in (
            (CARD, TEXT, "999"),
            ("", TEXT, "123"),
            (CARD, "No ebook markers", "123"),
            (CARD.replace("Public domain in the USA.", "Copyrighted."), TEXT, "123"),
            (CARD, TEXT.replace("Two friends", "Translation copyright 2018. Two friends"), "123"),
        ):
            with self.subTest(identifier=identifier, text=text):
                self.assertEqual(inspect_edition(card, text, identifier)["status"], "review")

    def test_cached_evidence_is_offline_and_checks_both_file_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "123.html").write_text(CARD)
            (root / "123.txt").write_text(TEXT)
            metadata = {"id": "123", "checked_at": utcnow().isoformat(),
                        "card_sha256": sha256(root / "123.html"),
                        "text_sha256": sha256(root / "123.txt")}
            atomic_json(root / "123.json", metadata)
            store = EditionStore(root)
            with patch("gutenberg.fetch_bytes", side_effect=AssertionError("Unexpected network request")):
                self.assertEqual(cached_edition(store, "123")["status"], "public_domain_us")
                for filename in ("123.html", "123.txt"):
                    path = root / filename
                    original = path.read_bytes()
                    path.write_text("Changed evidence")
                    self.assertIsNone(cached_edition(store, "123"))
                    path.write_bytes(original)
                metadata["checked_at"] = (utcnow() - timedelta(days=31)).isoformat()
                atomic_json(root / "123.json", metadata)
                self.assertIsNone(cached_edition(store, "123"))
                self.assertIsNone(cached_edition(store, "456"))

    def test_explicit_refresh_replaces_invalid_cache_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("gutenberg.fetch_bytes", side_effect=[CARD.encode(), TEXT.encode()]) as fetch:
                edition = EditionStore(root).load("123")
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(edition["status"], "public_domain_us")
            metadata = json.loads((root / "123.json").read_text())
            self.assertEqual(metadata["text_sha256"], sha256(root / "123.txt"))
            self.assertEqual(metadata["card_sha256"], sha256(root / "123.html"))

    def test_catalog_rejects_missing_columns_and_keeps_english_text_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.csv"
            path.write_text('Text#,Type,Title,Language,Authors\n123,Text,Example,en,"Writer, Ada"\n'
                            '456,Text,Other,fr,"Writer, Ada"\n')
            catalog = read_catalog(path)
            self.assertEqual(set(catalog), {"example"})
            self.assertEqual(catalog["example"]["123"]["author"], "Writer, Ada")
            path.write_text("Title\nExample\n")
            with self.assertRaises(ValueError):
                read_catalog(path)
