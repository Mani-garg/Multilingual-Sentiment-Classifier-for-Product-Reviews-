"""Dynamic micro-batching queue for combining concurrent inference requests."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class QueuedRequest:
    """Internal container representing an incoming item in the batching queue."""

    text: str
    language: str | None
    future: asyncio.Future
    created_at: float


class DynamicBatcher:
    """Micro-batching coordinator aggregating concurrent requests within a max wait interval."""

    def __init__(
        self,
        predict_fn: Callable[[list[str], list[str | None]], list[Any]],
        max_batch_size: int = 32,
        max_wait_ms: int = 20,
    ) -> None:
        self.predict_fn = predict_fn
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000.0
        self.queue: list[QueuedRequest] = []
        self.lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._is_running = True

    async def submit(self, text: str, language: str | None = None) -> Any:
        """Submit a single review to the batcher and await the micro-batch result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        req = QueuedRequest(
            text=text, language=language, future=future, created_at=time.perf_counter()
        )

        async with self.lock:
            self.queue.append(req)
            if len(self.queue) >= self.max_batch_size:
                # Reached batch capacity: trigger immediate processing
                batch = self.queue[: self.max_batch_size]
                self.queue = self.queue[self.max_batch_size :]
                asyncio.create_task(self._process_batch(batch))
            elif len(self.queue) == 1:
                # First element: schedule delayed flush
                if self._flush_task is None or self._flush_task.done():
                    self._flush_task = asyncio.create_task(self._delayed_flush())

        return await future

    async def _delayed_flush(self) -> None:
        """Wait for max_wait_seconds before flushing pending items."""
        await asyncio.sleep(self.max_wait_seconds)
        async with self.lock:
            if self.queue:
                batch = self.queue[: self.max_batch_size]
                self.queue = self.queue[self.max_batch_size :]
                asyncio.create_task(self._process_batch(batch))

    async def _process_batch(self, batch: list[QueuedRequest]) -> None:
        """Run batch inference on a thread pool and resolve each item's future."""
        if not batch:
            return

        texts = [r.text for r in batch]
        langs = [r.language for r in batch]

        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, self.predict_fn, texts, langs)

            for req, res in zip(batch, results, strict=False):
                if not req.future.done():
                    req.future.set_result(res)
        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    def shutdown(self) -> None:
        """Clean shutdown canceling pending tasks."""
        self._is_running = False
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
