"""ModelExpress tracing around the standard watcher and dispatcher behavior."""

import asyncio
from contextlib import ExitStack
from contextvars import ContextVar

from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.utils import min_fresh_version
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.utils.modelexpress_trace import CancellationTrace


class ModelExpressWeightWatcher(WeightWatcher):
    def __init__(self, receiver, **kwargs):
        super().__init__(receiver, **kwargs)
        self.update_lock = receiver.trace_lock("wait_update_lock")

    async def sync_startup(self, step: int, timeout: float) -> None:
        with self.receiver.trace_update(step), self.receiver.trace_phase("startup_update"):
            await super().sync_startup(step, timeout)

    async def apply_policy_update(self, next_step: int) -> None:
        with self.receiver.trace_update(next_step), self.receiver.trace_phase("watcher_update"):
            await super().apply_policy_update(next_step)

    async def _notify_update(self, step: int) -> None:
        with self.receiver.trace_phase(
            "update_notifications", {"observer.count": len(self.observers), "hook.count": len(self.update_hooks)}
        ):
            await super()._notify_update(step)

    def on_update(self, hook) -> None:
        async def traced_hook(step):
            with self.receiver.trace_phase("update_hook"):
                await hook(step)

        super().on_update(traced_hook)

    async def start(self) -> None:
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                with self.receiver.trace_phase("watcher_scan", {"wait.poll_interval_s": self.poll_interval}):
                    next_step = self.receiver.next_version(self.ckpt_step)
                if next_step > self.ckpt_step:
                    await self.apply_policy_update(next_step)
                with self.receiver.trace_phase("watcher_poll_sleep", {"wait.poll_interval_s": self.poll_interval}):
                    await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            return


class ModelExpressDispatcher(Dispatcher):
    def __init__(self, *, receiver, **kwargs):
        super().__init__(**kwargs)
        self.receiver = receiver
        self.scheduling_lock = receiver.trace_lock("wait_scheduling_lock")
        self._cancellation: ContextVar[CancellationTrace | None] = ContextVar("mx_stale_cancellation", default=None)

    async def on_version_pending(self, step: int) -> None:
        with self.receiver.trace_phase("pending_observer"), ExitStack() as stack:
            token = self._cancellation.set(CancellationTrace(stack, self.receiver.trace_phase))
            try:
                await super().on_version_pending(step)
            finally:
                self._cancellation.reset(token)

    async def drop_group(self, group_id, *, reason):
        cancellation = self._cancellation.get()
        if cancellation is None or reason != "stale":
            return await super().drop_group(group_id, reason=reason)
        if cancellation.attributes is None and self.progress is not None:
            cancellation.minimum = min_fresh_version(self.progress.step, self.max_off_policy_steps)
        return await cancellation.record_group(super().drop_group, group_id, reason)

    async def on_new_version(self, step: int) -> None:
        with self.receiver.trace_phase("new_version_observer"):
            await super().on_new_version(step)
