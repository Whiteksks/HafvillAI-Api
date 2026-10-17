"""Batch export selection, provenance, duplicate handling, and interrupted-run recovery."""

import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server import create_app
from summaries import RULES, ResumeSummary, StartSummary, SummaryManager, dialogue_text
from test_summaries import fixture, success


def manager_for(settings, text="Summary."):
    return SummaryManager(settings, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=success(text))))


async def generate(manager, batch, **settings):
    await manager.start(StartSummary(model="test-model", batch_number=batch, **settings))
    await manager.task
    assert manager.status()["status"] == "completed"
    return manager.run_dir()


class MergeTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_batches_last_short_batch_latest_duplicate_and_test_exclusion(self):
        with tempfile.TemporaryDirectory() as workspace:
            settings, source, rows, raw = fixture(Path(workspace), count=205)
            manager = manager_for(settings, "Original.")
            first = await generate(manager, 1)
            await generate(manager, 2)
            last = await generate(manager, 3)
            self.assertTrue(first.name.startswith("coser_batch_1_100_"))
            self.assertTrue(last.name.startswith("coser_batch_3_5_"))
            manager.transport = httpx.MockTransport(lambda request: httpx.Response(200, json=success("Replacement.")))
            replacement = await generate(manager, 1)
            test = await generate(manager, 1, limit=24)
            self.assertTrue(test.name.startswith("coser_batch_1_24_"))
            incomplete = await generate(manager, 2)
            # Same record count as a full batch, but a gap and an out-of-range position.
            with sqlite3.connect(incomplete / "results.sqlite3") as db:
                db.execute("UPDATE records SET position=205 WHERE position=150")
            await generate(manager, 0, limit=24)
            result = manager.merge()
            records = [json.loads(line) for line in manager.merged_artifact(result["id"]).read_text().splitlines()]
            self.assertEqual((result["records"], result["batches"]), (205, [1, 2, 3]))
            self.assertEqual([row["messages"][1]["content"] for row in records], [dialogue_text(row) for row in rows])
            self.assertTrue(all(row["messages"][2]["content"] == "Replacement." for row in records[:100]))
            self.assertEqual(source.read_bytes(), raw)
            report = json.loads((manager.merged_artifact(result["id"]).parent / "merge_report.json").read_text())
            self.assertEqual(report["batches"][0]["folder"], replacement.name)
            self.assertEqual({r["reason"] for r in report["excluded"]}, {"not_full_batch", "incomplete", "older_duplicate"})
            restored = SummaryManager(settings)
            restored.recover()
            self.assertEqual(restored.status()["latest_merge"], result)

    async def test_different_source_is_excluded_even_with_same_record_count(self):
        with tempfile.TemporaryDirectory() as workspace:
            settings, source, _, _ = fixture(Path(workspace), count=205)
            manager = manager_for(settings)
            await generate(manager, 1)
            lines = source.read_text().splitlines()
            row = json.loads(lines[0])
            row["messages"][1]["content"] = "Changed source dialogue."
            lines[0] = json.dumps(row)
            source.write_text("\n".join(lines) + "\n")
            await generate(manager, 2)
            result = manager.merge()
            self.assertEqual((result["records"], result["batches"]), (100, [2]))
            report = json.loads((manager.merged_artifact(result["id"]).parent / "merge_report.json").read_text())
            self.assertEqual(report["excluded"][0]["reason"], "different_source")

    async def test_mixed_rules_are_rejected_until_batch_is_regenerated(self):
        with tempfile.TemporaryDirectory() as workspace:
            settings, _, _, _ = fixture(Path(workspace), count=205)
            manager = manager_for(settings)
            await generate(manager, 1)
            await generate(manager, 2, rules="Different rules.")
            with self.assertRaises(HTTPException) as error:
                manager.merge()
            self.assertEqual(error.exception.status_code, 409)
            self.assertFalse((manager.root / "merged").exists())
            await generate(manager, 2, rules=RULES)
            self.assertEqual(manager.merge()["records"], 200)

    async def test_tests_and_incomplete_runs_are_not_full_batches(self):
        with tempfile.TemporaryDirectory() as workspace:
            settings, _, _, _ = fixture(Path(workspace), count=205)
            manager = manager_for(settings)
            await generate(manager, 1, limit=24)
            incomplete = await generate(manager, 2)
            with sqlite3.connect(incomplete / "results.sqlite3") as db:
                db.execute("DELETE FROM records WHERE position=150")
            with self.assertRaises(HTTPException) as error:
                manager.merge()
            self.assertEqual(error.exception.status_code, 409)

    async def test_interrupted_run_resume_preserves_saved_records(self):
        with tempfile.TemporaryDirectory() as workspace:
            settings, _, rows, _ = fixture(Path(workspace), count=205)
            manager = manager_for(settings)
            directory = await generate(manager, 3)
            with sqlite3.connect(directory / "results.sqlite3") as db:
                db.execute("DELETE FROM records WHERE position>=202")
            manager.update(status="running", completed=0)
            calls = []

            def handler(request):
                calls.append(json.loads(request.content)["input"][0]["content"])
                return httpx.Response(200, json=success())

            restored = SummaryManager(settings, transport=httpx.MockTransport(handler))
            restored.recover()
            self.assertTrue(directory.is_dir())
            self.assertEqual(restored.status()["completed"], 2)
            self.assertEqual(restored.status()["run_folder"], directory.name)
            await restored.resume(ResumeSummary())
            await restored.task
            self.assertEqual(calls, [dialogue_text(row) for row in rows[202:]])
            self.assertEqual(restored.merge()["records"], 5)

    async def test_corrupted_target_blocks_export(self):
        with tempfile.TemporaryDirectory() as workspace:
            settings, _, _, _ = fixture(Path(workspace), count=105)
            manager = manager_for(settings)
            directory = await generate(manager, 2)
            with sqlite3.connect(directory / "results.sqlite3") as db:
                db.execute("UPDATE records SET record='{}' WHERE position=100")
            with self.assertRaises(HTTPException) as error:
                manager.merge()
            self.assertEqual(error.exception.status_code, 422)


class MergeApiTests(unittest.TestCase):
    def test_merge_download_local_guard_and_reload(self):
        with tempfile.TemporaryDirectory() as workspace:
            settings, _, _, _ = fixture(Path(workspace), count=105)
            app = create_app(settings)
            calls = []

            def handler(request):
                calls.append(request)
                return httpx.Response(200, json=success())

            app.state.summaries.transport = httpx.MockTransport(handler)
            headers = {"X-Studio-Request": "1"}
            with TestClient(app) as client:
                self.assertEqual(client.post("/api/summaries/merge").status_code, 403)
                for batch in [1, 2]:
                    response = client.post("/api/summaries/start", json={"model": "test-model", "batch_number": batch}, headers=headers)
                    self.assertEqual(response.status_code, 202)
                    for _ in range(300):
                        status = client.get("/api/summaries/status").json()
                        if not status["busy"]:
                            break
                        time.sleep(.01)
                    self.assertEqual(status["status"], "completed")
                before = len(calls)
                response = client.post("/api/summaries/merge", headers=headers)
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(len(calls), before)
                self.assertEqual(result["records"], 105)
                download = client.get(f"/api/summaries/merged/{result['id']}")
                self.assertEqual(download.status_code, 200)
                self.assertEqual(len(download.text.splitlines()), 105)
                self.assertEqual(client.get("/api/summaries/merged/bad-id").status_code, 404)
            with TestClient(create_app(settings)) as client:
                self.assertEqual(client.get("/api/summaries/status").json()["latest_merge"], result)


if __name__ == "__main__":
    unittest.main()
