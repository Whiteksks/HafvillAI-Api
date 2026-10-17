import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from .schemas import GenerationFailed, InputTooLong

logger = logging.getLogger(__name__)


class Busy(Exception):
    pass


class Runtime:
    """One bounded queue and one GPU thread shared by both roles."""

    def __init__(self, engine, settings):
        self.engine = engine
        self.queue = asyncio.Queue(maxsize=settings.max_queue_size)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mamba")
        self.worker = None
        self.ready = False
        self.active = False

    async def start(self):
        await asyncio.get_running_loop().run_in_executor(self.executor, self.engine.load)
        self.worker = asyncio.create_task(self._work())
        self.ready = True

    async def generate(self, role, messages, max_new_tokens, request_id):
        if not self.ready:
            raise GenerationFailed("API не готов")
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((role, messages, max_new_tokens, request_id, future))
        except asyncio.QueueFull:
            raise Busy from None
        # Cancelling an HTTP request cancels its future; queued work is skipped.
        # Generation already in progress finishes to preserve serialized GPU access.
        return await future

    async def _work(self):
        loop = asyncio.get_running_loop()
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                role, messages, max_new_tokens, request_id, future = item
                if future.cancelled():
                    continue
                self.active = True
                try:
                    result = await loop.run_in_executor(
                        self.executor, self.engine.generate, role, messages, max_new_tokens, request_id,
                    )
                except InputTooLong as exc:
                    if not future.done():
                        future.set_exception(exc)
                except Exception:
                    logger.exception("Generation failed: request_id=%s role=%s", request_id, role)
                    if not future.done():
                        future.set_exception(GenerationFailed("Генерация завершилась ошибкой; подробности в терминале API"))
                else:
                    if not future.done():
                        future.set_result(result)
                finally:
                    self.active = False
            finally:
                self.queue.task_done()

    async def close(self):
        self.ready = False
        if self.worker is not None:
            await self.queue.put(None)
            await self.worker
        try:
            await asyncio.get_running_loop().run_in_executor(self.executor, self.engine.close)
        finally:
            self.executor.shutdown(wait=True)
