from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coser_catalog import POLICY as BOOK_POLICY, match_author
from coser_data import clean_utterances
from coser_sft import build, check, corrupt_scene, fingerprint
from coser_download import REVISION
from data_utils import sha256
from training.config import TrainConfig
from training.data import prepare
from training.tests.test_training import Tokenizer


def scene(subject):
    return {"scenario": "Two friends meet in a library.", "dialogues": [
        {"character": "Alex", "message": f"[I am curious.] Where is the {subject}?"},
        {"character": "Mara", "message": f"(nods) The {subject} is on the table."},
        {"character": "Alex", "message": "Thank you. May I read it?"},
        {"character": "Mara", "message": "Yes. Please return it tomorrow."}]}


class CoserSftTests(unittest.TestCase):
    def test_author_confirmation_rejects_missing_and_ambiguous_evidence(self):
        candidates = [{"id": "1", "author": "Austen, Jane, 1775-1817"},
                      {"id": "2", "author": "Bronte, Emily, 1818-1848"}]
        selected, _ = match_author(candidates, {"Mara": "A character from Jane Austen's novel."})
        self.assertEqual([c["id"] for c in selected], ["1"])
        for profiles in [{"Mara": "No author mentioned."}, {"Mara": "Jane Austen and Emily Bronte."}]:
            self.assertEqual(match_author(candidates, profiles)[0], [])

    def test_spoken_duplicates_ignore_aliases_and_added_actions(self):
        a = clean_utterances(scene("book"))
        alternate = scene("book")
        for u in alternate["dialogues"]:
            u["character"] = {"Alex": "John", "Mara": "Jane"}[u["character"]]
            u["message"] = "(smiles) " + u["message"]
        self.assertEqual(fingerprint(a), fingerprint(clean_utterances(alternate)))
        self.assertNotEqual(fingerprint(a), fingerprint(clean_utterances(scene("letter"))))

    def test_quality_filter_keeps_short_replies_but_rejects_source_loops(self):
        self.assertFalse(corrupt_scene([{ "speech": "Yes. Yes."}]))
        self.assertTrue(corrupt_scene([{ "speech": "We will return tomorrow. " * 15}]))

    def test_full_pipeline_never_trains_on_test_books(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            root = data / "coser"
            raw = root / "raw"
            (raw / "full").mkdir(parents=True)
            (raw / "test").mkdir()
            (data / "pg_catalog.csv").write_text("Fixture catalog\n")
            edition_dir = data / "gutenberg/editions"
            edition_dir.mkdir(parents=True)
            for suffix in ("html", "txt"):
                (edition_dir / f"1.{suffix}").write_text("Fixture evidence")
            editions = [{"id": "1", "status": "public_domain_us",
                         "card_sha256": sha256(edition_dir / "1.html"),
                         "text_sha256": sha256(edition_dir / "1.txt")}]
            files, entries = {}, []
            for name in ("A Fiction", "B Fiction", "T Fiction"):
                plots = [{"chapter": f"Chapter {i}", "key_characters": [],
                          "conversation": [scene(f"{name} book {i}"), scene("shared book")]} for i in range(2)]
                source = {"plots": plots, "character_datasets": {}, "split_plot_index": 1}
                path = raw / "full" / f"{name}.json"
                path.write_text(json.dumps(source))
                relative = f"full/{name}.json"
                files[relative] = {"sha256": sha256(path)}
                entries.append({"book": name, "file": relative, "status": "selected", "editions": editions,
                                "author_evidence": {"Example, Author": {"matched_name": "author example"}}})
            (raw / "test/ood_books_for_coser_models.txt").write_text("T Fiction\n")
            (root / "download_manifest.json").write_text(json.dumps({"revision": REVISION, "dataset": "Neph0s/CoSER", "files": files}))
            (root / "book_catalog.json").write_text(json.dumps({"policy": BOOK_POLICY, "revision": REVISION,
                "download_manifest_sha256": sha256(root / "download_manifest.json"),
                "pg_catalog_sha256": sha256(data / "pg_catalog.csv"), "books": entries}))
            config = TrainConfig("consciousness", str(data / "export/sft.jsonl"), str(data / "output"),
                                 source_dataset=str(data / "export/sft.jsonl"), prepared_dir=str(data / "tensors"),
                                 max_seq_length=2048, split_group_field="split_group")
            path = data / "config.json"
            path.write_text(json.dumps(config.as_dict()))
            with patch("coser_sft.load_tokenizer", return_value=Tokenizer()), \
                    patch("training.data.load_tokenizer", return_value=Tokenizer()), redirect_stdout(io.StringIO()):
                result = build(root, path)
                prepare(config, offline=True)
                checked = check(root, path)
            self.assertEqual(checked["status"], "passed")
            self.assertGreater(result["exclusions"]["duplicate_spoken_scene"], 0)
            self.assertGreater(result["exclusions"]["official_id_heldout_plot"], 0)
            self.assertEqual(checked["book_and_scene_overlap"], 0)
            self.assertEqual(result["statistics"]["train"]["books"], 1)
            self.assertEqual(result["statistics"]["validation"]["books"], 1)
            self.assertEqual(result["statistics"]["test"]["books"], 1)
            self.assertEqual(result["statistics"]["test"]["source_scenes"], 3)
            self.assertFalse(result["training_started"])
            with self.assertRaises(FileExistsError):
                build(root, path)
            with (data / "export/test.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                check(root, path)


if __name__ == "__main__":
    unittest.main()
