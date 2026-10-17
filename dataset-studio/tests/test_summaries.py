"""Summary pipeline checks with local dialogues and mocked OpenAI responses."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from config import Settings
from server import create_app
from summaries import (INSTRUCTION, RULES, ResumeSummary,
                       StartSummary, SummaryError, SummaryManager, dialogue_text,
                       response_text)


def fixture(directory, count=2):
    settings = Settings(data_dir=directory, env_file=None, openai_api_key="test-secret", summary_model="test-model")
    path = settings.data_dir / "coser/summary-v1/dialogues.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"messages": [{"role": "system", "content": "You are a librarian."},
                      {"role": "user", "content": "Alice: Please keep this letter.\nIt matters."},
                      {"role": "assistant", "content": "I will keep it safe."}]},
        {"messages": [{"role": "user", "content": "Bob: Did Alice return?"},
                      {"role": "assistant", "content": "She has not returned."}]},
    ]
    rows = rows[:count]
    rows.extend({"messages": [{"role": "user", "content": f"Dialogue {index + 1}: A visitor delivered a letter."}]}
                for index in range(len(rows), count))
    for index, row in enumerate(rows):
        row.update(example_id=f"scene-{index}", split="train" if index < 100 else "validation",
                   split_group="book-train" if index < 100 else "book-validation",
                   source={"dataset": "CoSER", "book": "Example", "conversation_index": index})
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    return settings, path, rows, raw


def success(text="Alice entrusted a letter to the librarian, who promised to keep it safe."):
    return {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
    ], "usage": {"input_tokens": 40, "output_tokens": 20}}


class ResponseTests(unittest.TestCase):
    def test_extracts_only_assistant_output_not_reasoning(self):
        self.assertEqual(response_text(success("  Summary.\n ")), "Summary.")

    def test_incomplete_refusal_empty_and_formatted_outputs_are_not_training_targets(self):
        responses = [
            {"status": "incomplete", "output": success()["output"]},
            {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "No"}]}]},
            success(""), success("```json\n{}\n```"), success('{"summary":"Example"}'),
        ]
        for response in responses:
            with self.subTest(response=response), self.assertRaises(SummaryError):
                response_text(response)


class SummaryJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_api_output_is_wrapped_locally_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, source, rows, raw = fixture(Path(directory))
            requests = []

            def handler(request):
                requests.append(request)
                return httpx.Response(200, json=success())

            manager = SummaryManager(settings, transport=httpx.MockTransport(handler))
            result = await manager.start(StartSummary(model="test-model", instruction="Summarize this conversation.", rules=RULES))
            self.assertTrue(result["busy"])
            with self.assertRaises(HTTPException) as context:
                await manager.start(StartSummary(model="test-model"))
            self.assertEqual(context.exception.status_code, 409)
            await manager.task
            self.assertEqual(manager.status()["status"], "completed")
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(len(requests), 2)
            payload = json.loads(requests[0].content)
            self.assertEqual(str(requests[0].url), "https://api.openai.com/v1/responses")
            self.assertEqual(requests[0].headers["authorization"], "Bearer test-secret")
            self.assertFalse(payload["store"])
            self.assertEqual(payload["input"], [{"role": "user", "content": dialogue_text(rows[0])}])
            self.assertIn(RULES, payload["instructions"])
            exported = [json.loads(line) for line in manager.artifact().read_text().splitlines()]
            self.assertEqual(exported[0], {**{key: rows[0][key] for key in ("example_id", "split", "split_group", "source")}, "messages": [
                {"role": "system", "content": "Summarize this conversation."},
                {"role": "user", "content": dialogue_text(rows[0])},
                {"role": "assistant", "content": success()["output"][1]["content"][0]["text"]},
            ]})
            self.assertEqual(manager.status()["input_tokens"], 80)
            self.assertNotIn("test-secret", json.dumps(manager.status()))
            for file in manager.root.rglob("*"):
                if file.is_file():
                    self.assertNotIn(b"test-secret", file.read_bytes())

    async def test_stop_and_restart_resume_saved_snapshot_without_repeating_paid_work(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, source, rows, _ = fixture(Path(directory))
            started, release = asyncio.Event(), asyncio.Event()
            calls = []

            async def handler(request):
                calls.append(json.loads(request.content)["input"][0]["content"])
                started.set()
                await release.wait()
                return httpx.Response(200, json=success("Saved first summary."))

            manager = SummaryManager(settings, transport=httpx.MockTransport(handler))
            await manager.start(StartSummary(model="test-model", concurrency=1))
            await started.wait()
            with self.assertRaises(HTTPException):
                manager.artifact()
            await manager.stop()
            release.set()
            await manager.task
            self.assertEqual(manager.status()["status"], "stopped")
            self.assertEqual(manager.status()["completed"], 1)
            self.assertEqual(len(manager.artifact().read_text().splitlines()), 1)
            # Progress in SQLite is authoritative even if the state file lagged.
            manager.update(status="running", completed=0)
            source.write_text('{"messages":[]}\n')
            def remaining(request):
                calls.append(json.loads(request.content)["input"][0]["content"])
                return httpx.Response(200, json=success("Saved second summary."))
            recovered = SummaryManager(settings, transport=httpx.MockTransport(remaining))
            recovered.recover()
            self.assertEqual(recovered.status()["status"], "interrupted")
            self.assertEqual(recovered.status()["completed"], 1)
            await recovered.resume(ResumeSummary())
            await recovered.task
            self.assertEqual(recovered.status()["status"], "completed")
            self.assertEqual(calls, [dialogue_text(row) for row in rows])
            self.assertEqual(len(recovered.artifact().read_text().splitlines()), 2)

    async def test_failure_preserves_partial_result_and_retries_only_unsaved_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, rows, _ = fixture(Path(directory))
            calls = []
            def handler(request):
                calls.append(json.loads(request.content)["input"][0]["content"])
                if len(calls) == 2:
                    return httpx.Response(401, json={"error": {"message": "test-secret"}})
                return httpx.Response(200, json=success())
            manager = SummaryManager(settings, transport=httpx.MockTransport(handler))
            await manager.start(StartSummary(model="test-model", concurrency=1))
            await manager.task
            self.assertEqual(manager.status()["status"], "failed")
            self.assertEqual(manager.status()["completed"], 1)
            self.assertNotIn("test-secret", manager.status()["error"])
            self.assertEqual(len(manager.artifact().read_text().splitlines()), 1)
            await manager.resume(ResumeSummary())
            await manager.task
            self.assertEqual(manager.status()["status"], "completed")
            self.assertEqual(calls, [dialogue_text(rows[0]), dialogue_text(rows[1]), dialogue_text(rows[1])])

    async def test_limit_quota_and_invalid_source_do_not_write_bad_records(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, source, _, _ = fixture(Path(directory))
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})
            manager = SummaryManager(settings, transport=httpx.MockTransport(handler))
            await manager.start(StartSummary(model="test-model", limit=1))
            await manager.task
            self.assertEqual(manager.status()["total"], 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(manager.status()["completed"], 0)
            with self.assertRaises(HTTPException):
                manager.artifact()
            source.write_text('{"messages":[]}\n')
            with self.assertRaises(HTTPException):
                await manager.start(StartSummary(model="test-model"))
            self.assertEqual(len(calls), 1)

    async def test_temporary_rate_limit_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, _, _ = fixture(Path(directory))
            calls = []
            def handler(request):
                calls.append(request)
                return (httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"code": "rate_limit_exceeded"}})
                        if len(calls) == 1 else httpx.Response(200, json=success()))
            manager = SummaryManager(settings, transport=httpx.MockTransport(handler))
            await manager.start(StartSummary(model="test-model", limit=1))
            await manager.task
            self.assertEqual(manager.status()["status"], "completed")
            self.assertEqual(len(calls), 2)

    async def test_batch_selection_four_concurrent_requests_and_ordered_export(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, source, rows, raw = fixture(Path(directory), count=205)
            started, release = asyncio.Event(), asyncio.Event()
            active = peak = 0
            calls, finished = [], []
            async def handler(request):
                nonlocal active, peak
                dialogue = json.loads(request.content)["input"][0]["content"]
                calls.append(dialogue)
                active += 1
                peak = max(peak, active)
                if active == 4:
                    started.set()
                await release.wait()
                # Ensure later responses can be saved before earlier ones.
                await asyncio.sleep(.001 if "Dialogue 104:" in dialogue else .005)
                active -= 1
                finished.append(dialogue)
                return httpx.Response(200, json=success("Summary of " + dialogue))
            manager = SummaryManager(settings, transport=httpx.MockTransport(handler))
            status = manager.status()
            self.assertEqual((status["source_records"], status["batch_count"]), (205, 3))
            await manager.start(StartSummary(model="test-model", batch_number=2, concurrency=4))
            await asyncio.wait_for(started.wait(), timeout=2)
            self.assertEqual(manager.status()["active_requests"], 4)
            self.assertEqual(manager.status()["total"], 100)
            release.set()
            await manager.task
            self.assertEqual(peak, 4)
            self.assertEqual(manager.status()["active_requests"], 0)
            self.assertEqual(manager.status()["status"], "completed")
            expected = [dialogue_text(row) for row in rows[100:200]]
            self.assertEqual(calls, expected)
            self.assertNotEqual(finished, expected)
            exported = [json.loads(line) for line in manager.artifact().read_text().splitlines()]
            self.assertEqual([row["messages"][1]["content"] for row in exported], expected)
            self.assertEqual(len(exported), 100)
            self.assertEqual(source.read_bytes(), raw)
            calls.clear()
            await manager.start(StartSummary(model="test-model", batch_number=3, concurrency=4))
            await manager.task
            self.assertEqual(manager.status()["total"], 5)
            self.assertEqual(calls, [dialogue_text(row) for row in rows[200:]])
            with self.assertRaises(HTTPException) as context:
                await manager.start(StartSummary(model="test-model", batch_number=4))
            self.assertEqual(context.exception.status_code, 422)

    async def test_concurrent_failure_drains_responses_and_resume_fills_only_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, rows, _ = fixture(Path(directory), count=6)
            texts = [dialogue_text(row) for row in rows]
            started, release = asyncio.Event(), asyncio.Event()
            calls = []
            async def handler(request):
                dialogue = json.loads(request.content)["input"][0]["content"]
                calls.append(dialogue)
                if len(calls) == 4:
                    started.set()
                await release.wait()
                if dialogue == texts[1]:
                    return httpx.Response(401, json={"error": {"message": "No access"}})
                await asyncio.sleep(.01)
                return httpx.Response(200, json=success("Summary of " + dialogue))
            manager = SummaryManager(settings, transport=httpx.MockTransport(handler))
            await manager.start(StartSummary(model="test-model", concurrency=4))
            await asyncio.wait_for(started.wait(), timeout=2)
            release.set()
            await manager.task
            self.assertEqual(manager.status()["status"], "failed")
            self.assertEqual(manager.status()["completed"], 3)
            self.assertEqual(manager.status()["active_requests"], 0)
            self.assertEqual(calls, texts[:4])
            exported = [json.loads(line) for line in manager.artifact().read_text().splitlines()]
            self.assertEqual([row["messages"][1]["content"] for row in exported], [texts[i] for i in (0, 2, 3)])
            manager.update(status="running", completed=0)
            def remaining(request):
                dialogue = json.loads(request.content)["input"][0]["content"]
                calls.append(dialogue)
                return httpx.Response(200, json=success("Summary of " + dialogue))
            recovered = SummaryManager(settings, transport=httpx.MockTransport(remaining))
            recovered.recover()
            await recovered.resume(ResumeSummary())
            await recovered.task
            self.assertEqual(calls[4:], [texts[i] for i in (1, 4, 5)])
            exported = [json.loads(line) for line in recovered.artifact().read_text().splitlines()]
            self.assertEqual([row["messages"][1]["content"] for row in exported], texts)
            self.assertEqual(recovered.status()["status"], "completed")

    async def test_stop_finishes_active_requests_and_does_not_schedule_more(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, rows, _ = fixture(Path(directory), count=10)
            started, release = asyncio.Event(), asyncio.Event()
            calls = []
            async def handler(request):
                calls.append(json.loads(request.content)["input"][0]["content"])
                if len(calls) == 4:
                    started.set()
                await release.wait()
                return httpx.Response(200, json=success())
            manager = SummaryManager(settings, transport=httpx.MockTransport(handler))
            await manager.start(StartSummary(model="test-model", concurrency=4))
            await asyncio.wait_for(started.wait(), timeout=2)
            await manager.stop()
            release.set()
            await manager.task
            self.assertEqual(len(calls), 4)
            self.assertEqual(manager.status()["completed"], 4)
            self.assertEqual(manager.status()["status"], "stopped")
            await manager.resume(ResumeSummary())
            await manager.task
            self.assertEqual(calls, [dialogue_text(row) for row in rows])
            self.assertEqual(manager.status()["completed"], 10)


class SummaryApiTests(unittest.TestCase):
    def test_local_request_guard_connection_start_download_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, _, _ = fixture(Path(directory))
            app = create_app(settings)
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(200, json={"id": "test-model"} if request.method == "GET" else success())
            app.state.summaries.transport = httpx.MockTransport(handler)
            headers = {"X-Studio-Request": "1"}
            with TestClient(app) as client:
                status = client.get("/api/summaries/status").json()
                self.assertTrue(status["source_ready"])
                self.assertTrue(status["key_configured"])
                self.assertNotIn("test-secret", json.dumps(status))
                self.assertEqual(client.post("/api/summaries/start", json={"model": "test-model"}).status_code, 403)
                self.assertEqual(client.post("/api/summaries/check", json={"model": "test-model"}, headers=headers).status_code, 200)
                self.assertEqual(calls[0].method, "GET")
                self.assertEqual(client.post("/api/summaries/check", json={"model": "../bad"}, headers=headers).status_code, 422)
                result = client.post("/api/summaries/start", json={"model": "test-model", "limit": 1}, headers=headers)
                self.assertEqual(result.status_code, 202)
                import time
                for _ in range(200):
                    status = client.get("/api/summaries/status").json()
                    if not status["busy"]:
                        break
                    time.sleep(.01)
                self.assertEqual(status["status"], "completed")
                download = client.get("/api/summaries/download")
                self.assertEqual(download.status_code, 200)
                record = json.loads(download.text)
                self.assertEqual([m["role"] for m in record["messages"]], ["system", "user", "assistant"])
                self.assertEqual(record["messages"][0]["content"], INSTRUCTION)
            with TestClient(create_app(settings)) as client:
                self.assertEqual(client.get("/api/summaries/status").json()["completed"], 1)


if __name__ == "__main__":
    unittest.main()
