"""Clean scene preparation and summary provenance, with no paid API requests."""

import json
from pathlib import Path
import tempfile
import unittest

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from coser_summaries import prepare_source
from data_utils import sha256
from server import create_app
from summaries import ResumeSummary, StartSummary, SummaryError, SummaryManager, response_text
from test_summaries import fixture, success


def coser_fixture(root, count=105):
    settings, path, rows, _ = fixture(root, count=count)
    return settings, path, rows


class PreparationTests(unittest.TestCase):
    def test_full_scene_once_without_thoughts_environment_or_test_books(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            book = root / "raw/full/book.json"
            book.parent.mkdir(parents=True)
            scene = {"dialogues": [
                {"character": "Alice", "message": "[Private plan] Please keep my letter. (hands it over)"},
                {"character": "Environment", "message": "[Unobserved description]"},
                {"character": "Bob", "message": "[Private fear] I promise to keep it safe."},
                {"character": "Alice", "message": "I will return on Friday."}]}
            book.write_text(json.dumps({"plots": [{"conversation": [scene]}]}))
            (root / "download_manifest.json").write_text(json.dumps({"revision": "revision-1",
                "files": {"full/book.json": {"sha256": sha256(book)}}}))
            source = {"file": "full/book.json", "dataset": "CoSER", "revision": "revision-1",
                      "plot_index": 0, "conversation_index": 0, "book": "Book", "scene_sha256": "scene-1"}
            rows = [{"split": "train", "split_group": "book-1", "source": source},
                    {"split": "train", "split_group": "book-1", "source": dict(source, character="Bob")},
                    {"split": "test", "split_group": "test-book", "source": {}}]
            (root / "sft-v1").mkdir()
            (root / "sft-v1/sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = prepare_source(root)
            self.assertEqual(report["records"], 1)
            row = json.loads((root / "summary-v1/dialogues.jsonl").read_text())
            text = row["messages"][0]["content"]
            self.assertIn("Alice: I will return on Friday.", text)
            self.assertIn("(hands it over)", text)
            self.assertNotIn("Private", text)
            self.assertNotIn("Environment", text)
            self.assertEqual(row["split"], "train")
            book.write_text("{}")
            with self.assertRaises(ValueError):
                prepare_source(root)


class CoSERJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_batches_merge_source_isolation_metadata_and_default_model(self):
        with tempfile.TemporaryDirectory() as workspace:
            settings, source, rows = coser_fixture(Path(workspace))
            calls = []
            def handler(request):
                calls.append(json.loads(request.content))
                return httpx.Response(200, json=success())
            manager = SummaryManager(settings, transport=httpx.MockTransport(handler))
            self.assertEqual(manager.status()["source_kind"], "coser")
            self.assertEqual(manager.status()["defaults"]["model"], "test-model")
            await manager.start(StartSummary(model="test-model", batch_number=1))
            await manager.task
            for batch in (1, 2):
                await manager.start(StartSummary(model="gpt-6.1-sol", source_kind="coser", batch_number=batch))
                await manager.task
                self.assertEqual(manager.status()["status"], "completed")
            self.assertEqual(manager.state["completed"], 5)
            self.assertTrue(manager.run_dir().name.startswith("coser_batch_2_5_"))
            self.assertEqual(manager.artifact().name, "CoSER_summaries.jsonl")
            count = len(calls)
            merged = manager.merge("coser")
            self.assertEqual(len(calls), count)
            records = [json.loads(r) for r in manager.merged_artifact(merged["id"]).read_text().splitlines()]
            self.assertEqual(len(records), 105)
            self.assertEqual(records[100]["split"], "validation")
            self.assertEqual([r["example_id"] for r in records], [r["example_id"] for r in rows])
            self.assertTrue(all(c["model"] == "gpt-6.1-sol" for c in calls[100:]))
            self.assertEqual(manager.merge()["records"], 105)
            source.write_text(source.read_text() + source.read_text().splitlines()[0] + "\n")
            with self.assertRaises(HTTPException):
                manager.merge("coser")

    async def test_recovery_uses_saved_clean_snapshot_and_rejects_dirty_sources(self):
        with tempfile.TemporaryDirectory() as workspace:
            settings, source, _ = coser_fixture(Path(workspace), count=2)
            calls = []
            def fail(request):
                calls.append(request)
                return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})
            manager = SummaryManager(settings, transport=httpx.MockTransport(fail))
            await manager.start(StartSummary(model="gpt-6.1-sol", source_kind="coser", concurrency=1))
            await manager.task
            source.write_text('{"messages":[{"role":"user","content":"<think>bad</think>"}]}\n')
            restored = SummaryManager(settings, transport=httpx.MockTransport(lambda req: httpx.Response(200, json=success())))
            restored.recover()
            await restored.resume(ResumeSummary())
            await restored.task
            self.assertEqual(restored.state["completed"], 2)
            self.assertEqual(json.loads(restored.artifact().read_text().splitlines()[0])["example_id"], "scene-0")
            restored.transport = httpx.MockTransport(fail)
            before = len(calls)
            with self.assertRaises(HTTPException):
                await restored.start(StartSummary(model="gpt-6.1-sol", source_kind="coser"))
            self.assertEqual(len(calls), before)

    def test_reasoning_markup_is_not_a_training_target(self):
        for text in ("<think>Plan</think>Summary", "<system_thinking>Plan</system_thinking>", "<|eot_id|>"):
            with self.subTest(text=text), self.assertRaises(SummaryError):
                response_text(success(text))


class SourceApiTests(unittest.TestCase):
    def test_source_selection_and_invalid_source(self):
        with tempfile.TemporaryDirectory() as workspace:
            settings, _, _ = coser_fixture(Path(workspace))
            with TestClient(create_app(settings)) as client:
                self.assertEqual(client.get("/api/summaries/status").json()["source_kind"], "coser")
                self.assertEqual(client.get("/api/summaries/status?source_kind=her").status_code, 422)
                self.assertEqual(client.get("/api/her/status").status_code, 404)
                self.assertEqual(client.post("/api/summaries/start", json={"model": "test-model", "source_kind": "her"}, headers={"X-Studio-Request": "1"}).status_code, 422)
                self.assertEqual(client.get("/api/summaries/status?source_kind=unknown").status_code, 422)


if __name__ == "__main__":
    unittest.main()
