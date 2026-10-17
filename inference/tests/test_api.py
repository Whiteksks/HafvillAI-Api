import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from inference.api.app import create_app
from inference.api.runtime import Runtime
from inference.api.schemas import GenerateResponse, InputTooLong, Usage
from inference.api.settings import Settings


class FakeEngine:
    """Exercise HTTP handling and queueing; real CUDA inference is checked separately."""

    def __init__(self, settings):
        self.settings = settings
        self.steps = {}
        self.loads = 0
        self.closed = False
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.fail = False

    def load(self):
        self.loads += 1
        self.steps = {"consciousness": 650}
        if self.settings.enable_subconsciousness:
            self.steps["subconsciousness"] = 242

    def generate(self, role, messages, max_new_tokens, request_id):
        self.calls.append((role, messages, max_new_tokens))
        self.started.set()
        if self.block and not self.release.wait(5):
            raise RuntimeError("Test release timeout")
        if self.fail:
            raise RuntimeError("internal paths must not leak")
        if messages[-1]["content"] == "too long":
            raise InputTooLong("Вход превышает лимит; текст не обрезан")
        return GenerateResponse(
            request_id=request_id, role=role, adapter_step=self.steps[role], content="Hello.",
            finish_reason="eot", usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            generation_seconds=0.01,
        )

    def close(self):
        self.closed = True


def payload(text="Hello", **kwargs):
    return {"messages": [{"role": "user", "content": text}], "max_new_tokens": 8, **kwargs}


@asynccontextmanager
async def test_client(app):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield client


class APITests(unittest.IsolatedAsyncioTestCase):
    def make_app(self, **kwargs):
        kwargs.setdefault("enable_subconsciousness", False)
        return create_app(Settings(**kwargs), FakeEngine)

    async def test_load_once_response_contract_and_cleanup(self):
        app = self.make_app()
        engine = app.state.runtime.engine
        async with test_client(app) as client:
            health = (await client.get("/health")).json()
            self.assertEqual(health["adapter_steps"], {"consciousness": 650})
            for _ in range(2):
                response = (await client.post("/chat", json=payload())).json()
                self.assertEqual(response["content"], "Hello.")
                self.assertEqual(response["usage"]["total_tokens"], 12)
                self.assertEqual(response["adapter_step"], 650)
                self.assertEqual(len(response["request_id"]), 32)
            self.assertEqual(engine.loads, 1)
            self.assertEqual((await client.post("/summarize", json=payload())).status_code, 503)
        self.assertTrue(engine.closed)

    async def test_both_roles_route_to_their_own_adapter(self):
        app = self.make_app(enable_subconsciousness=True)
        async with test_client(app) as client:
            self.assertEqual((await client.post("/chat", json=payload())).json()["adapter_step"], 650)
            self.assertEqual((await client.post("/summarize", json=payload())).json()["adapter_step"], 242)
        self.assertEqual([call[0] for call in app.state.runtime.engine.calls], ["consciousness", "subconsciousness"])

    async def test_default_profile_enables_both_coser_roles_with_safe_limits(self):
        app = create_app(Settings(), FakeEngine)
        async with test_client(app) as client:
            health = (await client.get("/health")).json()
            self.assertEqual(health["adapter_steps"], {"consciousness":650,"subconsciousness":242})
            self.assertEqual(health["limits"]["input_tokens"] + health["limits"]["new_tokens"], 2048)
        self.assertEqual(Settings().subconsciousness_repetition_penalty, 1.2)

    async def test_oversized_context_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            Settings(max_input_tokens=2048)

    async def test_authentication(self):
        async with test_client(self.make_app(api_key="test-key")) as client:
            self.assertEqual((await client.get("/health")).status_code, 200)
            for headers in ({}, {"Authorization": "Bearer wrong"}):
                self.assertEqual((await client.post("/chat", json=payload(), headers=headers)).status_code, 401)
            self.assertEqual((await client.post("/chat", json=payload(), headers={"Authorization": "Bearer test-key"})).status_code, 200)

    async def test_validation_prevents_generation(self):
        app = self.make_app(max_new_tokens=16)
        invalid = [
            {"messages": []},
            {"messages": [{"role": "assistant", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "system", "content": "Hi"},
                          {"role": "user", "content": "Hi"}]},
            payload(max_new_tokens=True), payload(max_new_tokens=17), payload(extra="unknown"),
        ]
        async with test_client(app) as client:
            for body in invalid:
                with self.subTest(body=body):
                    self.assertEqual((await client.post("/chat", json=body)).status_code, 422)
        self.assertEqual(app.state.runtime.engine.calls, [])

    async def test_input_limit_and_model_error_are_distinct_and_worker_recovers(self):
        app = self.make_app()
        async with test_client(app) as client:
            self.assertEqual((await client.post("/chat", json=payload("too long"))).status_code, 413)
            app.state.runtime.engine.fail = True
            with self.assertLogs("inference.api.runtime", level="ERROR"):
                response = await client.post("/chat", json=payload())
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("internal paths", response.text)
            app.state.runtime.engine.fail = False
            self.assertEqual((await client.post("/chat", json=payload())).status_code, 200)

    async def test_startup_failure_cleans_up(self):
        class FailingEngine(FakeEngine):
            def load(self):
                raise RuntimeError("Failed to load")

        app = create_app(Settings(), FailingEngine)
        with self.assertRaisesRegex(RuntimeError, "Failed to load"):
            async with app.router.lifespan_context(app):
                pass
        self.assertTrue(app.state.runtime.engine.closed)


class EngineTests(unittest.TestCase):
    def test_cycle_stops_and_api_returns_one_copy_with_actual_usage(self):
        import torch
        from inference.api.engine import MambaEngine
        engine = MambaEngine(Settings())
        serializer = SimpleNamespace(
            eot=4, prompt=lambda messages: [6],
            tokenizer=SimpleNamespace(eos_token_id=5, decode=lambda tokens, **kw: 'x'*len(tokens)))
        engine.serializers['consciousness'] = serializer
        engine.steps['consciousness'] = 650
        logits = torch.tensor([[[0., 10., 0., 0., 0., 0., 0.]]])
        engine.models['consciousness'] = lambda *args, **kw: SimpleNamespace(logits=logits)
        tensor = torch.tensor
        def cpu_tensor(data, **kwargs):
            kwargs['device'] = 'cpu'
            return tensor(data, **kwargs)
        with patch('inference.api.engine.torch.tensor', side_effect=cpu_tensor), \
             patch('inference.api.engine.torch.cuda.synchronize'):
            result = engine.generate('consciousness', [{'role':'user','content':'test'}], 512, 'test')
        self.assertEqual(result.finish_reason, 'repetition')
        self.assertEqual(result.content, 'x')
        self.assertEqual(result.usage.completion_tokens, 128)


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_streamed_body_size_is_limited(self):
        app = create_app(Settings(max_body_bytes=80), FakeEngine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                async def body():
                    yield b"x" * 50
                    yield b"x" * 50

                response = await client.post("/chat", content=body(), headers={"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 413)
                self.assertEqual(app.state.runtime.engine.calls, [])

    async def test_queue_overload_and_health_while_gpu_is_busy(self):
        app = create_app(Settings(max_queue_size=1), FakeEngine)
        engine = app.state.runtime.engine
        engine.block = True
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/chat", json=payload("first")))
                second = None
                try:
                    self.assertTrue(await asyncio.to_thread(engine.started.wait, 2))
                    second = asyncio.create_task(client.post("/chat", json=payload("second")))
                    async with asyncio.timeout(2):
                        while app.state.runtime.queue.qsize() != 1:
                            await asyncio.sleep(0)
                    health = await client.get("/health")
                    self.assertTrue(health.json()["active"])
                    response = await client.post("/chat", json=payload("third"))
                    self.assertEqual(response.status_code, 429)
                    self.assertEqual(response.headers["Retry-After"], "5")
                finally:
                    engine.release.set()
                    await first
                    if second is not None:
                        await second
        self.assertEqual([call[1][-1]["content"] for call in engine.calls], ["first", "second"])

    async def test_cancellation_does_not_start_parallel_gpu_generation(self):
        settings = replace(Settings(), max_queue_size=2)
        engine = FakeEngine(settings)
        engine.block = True
        runtime = Runtime(engine, settings)
        await runtime.start()
        tasks = []
        try:
            first = asyncio.create_task(runtime.generate("consciousness", payload("first")["messages"], 8, "1"))
            tasks.append(first)
            self.assertTrue(await asyncio.to_thread(engine.started.wait, 2))
            second = asyncio.create_task(runtime.generate("consciousness", payload("second")["messages"], 8, "2"))
            tasks.append(second)
            async with asyncio.timeout(2):
                while runtime.queue.qsize() != 1:
                    await asyncio.sleep(0)
            first.cancel()
            second.cancel()
            for task in tasks:
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertTrue(runtime.active)
            third = asyncio.create_task(runtime.generate("consciousness", payload("third")["messages"], 8, "3"))
            tasks.append(third)
            await asyncio.sleep(0)
            self.assertEqual(len(engine.calls), 1)
            engine.release.set()
            self.assertEqual((await third).content, "Hello.")
        finally:
            engine.release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await runtime.close()
        self.assertEqual([call[1][-1]["content"] for call in engine.calls], ["first", "third"])


if __name__ == "__main__":
    unittest.main()
