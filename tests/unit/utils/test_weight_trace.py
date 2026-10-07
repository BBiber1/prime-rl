import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.utils.weight_trace import (
    Phase,
    PhaseRecorder,
    clock_identity,
    discovery_phases,
    publish_offer,
    read_offer,
)

CARRIER = {"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}


def test_offer_marker_is_atomic_and_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "mx.experiment.run_id=test-run")
    marker = tmp_path / ".sender_ready"
    publish_offer(marker, 3, CARRIER)
    offer = read_offer(marker, 3)
    assert offer["trace_context"] == CARRIER and offer["run_id"] == "test-run"
    assert not marker.with_suffix(".pending").exists()
    with pytest.raises(ValueError, match="step/run"):
        read_offer(marker, 4)
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "mx.experiment.run_id=other-run")
    with pytest.raises(ValueError, match="step/run"):
        read_offer(marker, 3)
    marker.write_text("")
    assert read_offer(marker, 3) is None
    marker.unlink()
    assert read_offer(marker, 3) is None


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"step": True},
        {"trace_context": {"traceparent": "malformed"}},
        {"trace_context": {"traceparent": "00-" + "0" * 32 + "-" + "2" * 16 + "-01"}},
        {"trace_context": []},
        {"offered_at_monotonic_ns": -1},
        {"sender_host": 1},
    ],
)
def test_invalid_offer_is_rejected(tmp_path, mutation):
    marker = tmp_path / ".sender_ready"
    publish_offer(marker, 3, CARRIER)
    offer = json.loads(marker.read_text())
    marker.write_text(json.dumps({**offer, **mutation}))
    with pytest.raises(ValueError):
        read_offer(marker, 3)


def test_discovery_clips_idle_sleep_only_for_shared_clock():
    poll = {
        "watcher_poll_sleep": Phase("watcher_poll_sleep", 100, 200, 10, 110, {}),
        "watcher_scan": Phase("watcher_scan", 200, 220, 110, 130, {}),
    }
    offer = {"sender_host": "host", "clock_id": "boot/ns", "offered_at_monotonic_ns": 60}
    phases = discovery_phases(offer, poll, ("host", "boot/ns"))
    sleep, scan, elapsed = phases
    assert (sleep.start, sleep.end) == (150, 200)
    assert (scan.start, scan.end) == (200, 220)
    assert (elapsed.start, elapsed.end) == (150, 220)
    for identity in (("other-host", "boot/ns"), ("host", "other-ns"), ("host", "")):
        phases = discovery_phases(offer, poll, identity)
        assert [p.name for p in phases] == ["watcher_scan"]
        assert phases[0].attributes["offer.clock_shared"] is False
        assert phases[0].attributes["watcher.previous_sleep_s"] == 100 / 1e9
    assert [p.name for p in discovery_phases({**offer, "offered_at_monotonic_ns": 140}, poll, ("host", "boot/ns"))] == [
        "watcher_scan"
    ]
    assert discovery_phases(offer, {}, clock_identity()) == []


@pytest.mark.parametrize("failure", [RuntimeError("failed"), asyncio.CancelledError()])
def test_phase_error_buffer_and_context_cleanup(failure):
    recorder = PhaseRecorder(lambda: True)
    phases = []
    with recorder.session():
        with pytest.raises(type(failure)), recorder.phase("prepare"):
            raise failure
        recorder.bind(phases.append)
        with recorder.phase("notify", {"count": 2}):
            pass
    assert phases[0].error is failure
    assert phases[0].start <= phases[0].end
    assert phases[1].attributes == {"count": 2}
    assert recorder.current.get() is None


def test_phase_buffers_are_bounded_and_disabled_is_inert():
    recorder = PhaseRecorder(lambda: True)
    phases = []
    with recorder.session() as session:
        for _ in range(50):
            with recorder.phase("prepare"):
                pass
        assert len(session.pending) == 32
        recorder.bind(phases.append)
    assert phases[0].attributes["trace.dropped_phases"] == 18
    for _ in range(50):
        for name in ("watcher_scan", "watcher_poll_sleep"):
            with recorder.phase(name):
                pass
    assert len(recorder.poll) == 2
    disabled = PhaseRecorder(lambda: False)
    with disabled.session() as session, disabled.phase("prepare"):
        pass
    assert not session.pending and not disabled.poll


@pytest.mark.asyncio
async def test_watcher_lock_races_notifications_and_cancellation():
    recorder = PhaseRecorder(lambda: True)
    phases, calls = [], []
    accepted = 0

    @contextmanager
    def update(step):
        with recorder.session():
            yield

    def accept():
        nonlocal accepted
        accepted += 1
        recorder.bind(phases.append)

    async def published(step, **kwargs):
        calls.append("offer")

    async def receive(step):
        calls.append("receive")

    async def pending(step):
        calls.append("pending")

    async def notified(step):
        calls.append("notify")

    async def hook(step):
        calls.append("hook")

    receiver = SimpleNamespace(
        trace_update=update, trace_accept=accept, trace_phase=recorder.phase, wait_published=published, receive=receive
    )
    watcher = WeightWatcher(
        receiver,
        policy=SimpleNamespace(version=0),
        observers=[SimpleNamespace(on_version_pending=pending, on_new_version=notified)],
    )
    watcher.on_update(hook)
    await watcher.update_lock.acquire()
    task = asyncio.create_task(watcher.apply_policy_update(1))
    await asyncio.sleep(0)
    assert calls == []
    watcher.update_lock.release()
    await task
    await watcher.apply_policy_update(1)
    assert accepted == 1
    assert calls == ["offer", "pending", "receive", "notify", "hook"]
    assert watcher.policy.version == 1 and watcher.update_count == 1
    assert [p.name for p in phases] == [
        "wait_update_lock",
        "confirm_offer",
        "pending_observer",
        "pending_observers",
        "policy_advance",
        "new_version_observer",
        "update_hook",
        "update_notifications",
    ]
    assert phases[-1].end >= phases[-2].end
    await watcher.update_lock.acquire()
    task = asyncio.create_task(watcher.apply_policy_update(2))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert accepted == 1 and watcher.update_lock.locked()
    watcher.update_lock.release()
    assert recorder.current.get() is None


@pytest.mark.asyncio
async def test_startup_notifications_and_default_polling(monkeypatch):
    recorder = PhaseRecorder(lambda: True)
    phases, calls = [], []

    @contextmanager
    def update(step):
        with recorder.session():
            yield

    async def startup(step, timeout):
        calls.append(("startup", step, timeout))

    async def notified(step):
        calls.append(("notify", step))

    receiver = SimpleNamespace(
        trace_update=update,
        trace_accept=lambda: recorder.bind(phases.append),
        trace_phase=recorder.phase,
        sync_startup=startup,
        next_version=lambda current: current,
    )
    watcher = WeightWatcher(
        receiver, policy=SimpleNamespace(version=0), observers=[SimpleNamespace(on_new_version=notified)]
    )
    await watcher.sync_startup(3, timeout=10)
    assert calls == [("startup", 3, 10), ("notify", 3)]
    assert watcher.policy.version == watcher.ckpt_step == 3
    assert [p.name for p in phases] == [
        "wait_update_lock",
        "policy_advance",
        "new_version_observer",
        "update_notifications",
    ]

    async def sleep(interval):
        calls.append(("sleep", interval))
        watcher.stopped.set()

    monkeypatch.setattr(asyncio, "sleep", sleep)
    await watcher.start()
    assert calls[-1] == ("sleep", 1.0)
    assert set(recorder.poll) == {"watcher_scan", "watcher_poll_sleep"}
    assert recorder.poll["watcher_scan"].attributes["wait.poll_interval_s"] == 1.0
