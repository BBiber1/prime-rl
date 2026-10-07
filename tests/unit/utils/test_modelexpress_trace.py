import asyncio
import os
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

pytest.importorskip("modelexpress")
pytest.importorskip("opentelemetry.sdk")

from modelexpress import telemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from prime_rl.utils.modelexpress_trace import CancellationTrace, ReceiverTracing, RefitLock
from prime_rl.utils.weight_trace import publish_offer


@pytest.fixture
def refit_trace(tmp_path, monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(telemetry.RefitSpanProcessor(SimpleSpanProcessor(exporter)))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused/v1/traces")
    monkeypatch.setattr(telemetry, "_configured_pid", os.getpid())
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("receiver-test"))
    root = telemetry.RefitCycle({"refit.root": True, "step": 3})
    carrier = {}
    root.inject(carrier)
    step_dir = tmp_path / "step_3"
    step_dir.mkdir()
    warnings = []
    receiver = ReceiverTracing(lambda step: step_dir, "H2D", SimpleNamespace(warning=warnings.append))
    yield receiver, root, carrier, step_dir, exporter, warnings
    root.finish()
    provider.shutdown()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.asyncio
async def test_receiver_envelope_includes_notifications_without_context_leaks(refit_trace, legacy):
    receiver, root, carrier, step_dir, exporter, _ = refit_trace
    marker = step_dir / ".sender_ready"
    if legacy:
        marker.touch()
    else:
        publish_offer(marker, 3, carrier)
    with receiver.trace_phase("watcher_scan"):
        pass
    contexts = []

    async def callback(step):
        async def rollout():
            contexts.append(trace.get_current_span().get_span_context().is_valid)
            assert not telemetry._refit_attributes.get()

        await asyncio.create_task(rollout())

    async def receive(step):
        receiver.trace_accept()
        with receiver.trace_phase("receiver_ack"):
            pass
        with receiver.trace_phase("wait_version_marker"):
            pass
        receiver.current.uid = "v3"
        if legacy:
            with receiver.trace_phase("version_context_lookup"):
                pass
            receiver.current.carrier = carrier
            receiver.trace_accept()
        receiver.current.role.set_attributes({"version_uid": "v3", "refit.id": "v3"})

    with receiver.trace_update(3), receiver.trace_phase("watcher_update"):
        receiver.trace_accept()
        await callback(3)
        await receive(3)
        with receiver.trace_phase("update_notifications"):
            await callback(3)
            with receiver.trace_phase("update_hook"):
                await callback(3)
    # A duplicate call that is not accepted exports no additional envelope.
    with receiver.trace_update(3), receiver.trace_phase("watcher_update"):
        pass
    root.finish()
    spans = exporter.get_finished_spans()
    roles = [s for s in spans if s.name == "mx.refit.orchestrator"]
    assert len(roles) == 1
    role = roles[0]
    children = [s for s in spans if s.parent is not None and s.parent.span_id == role.context.span_id]
    assert role.parent.span_id == next(s for s in spans if s.name == "mx.refit.cycle").context.span_id
    assert all(s.context.trace_id == role.context.trace_id for s in children)
    assert (role.start_time, role.end_time) == (min(s.start_time for s in children), max(s.end_time for s in children))
    assert any(s.name == "mx.refit.update_hook" for s in children)
    assert any(s.name == "mx.refit.receiver_ack" for s in children)
    assert contexts == [False, False, False]
    assert receiver.current is None and receiver._phases.current.get() is None


@pytest.mark.parametrize(
    "phase",
    [
        "wait_update_lock",
        "confirm_offer",
        "receiver_ack",
        "wait_version_marker",
        "version_context_lookup",
        "update_hook",
    ],
)
@pytest.mark.parametrize("failure", [RuntimeError("phase failed"), asyncio.CancelledError()])
def test_early_failure_is_correlated_and_cleans_up(refit_trace, phase, failure):
    receiver, root, carrier, step_dir, exporter, _ = refit_trace
    publish_offer(step_dir / ".sender_ready", 3, carrier)
    with pytest.raises(type(failure)), receiver.trace_update(3):
        if phase != "wait_update_lock":
            receiver.trace_accept()
        with receiver.trace_phase(phase):
            raise failure
    root.finish()
    spans = exporter.get_finished_spans()
    child = next(s for s in spans if s.name == f"mx.refit.{phase}")
    root_span = next(s for s in spans if s.name == "mx.refit.cycle")
    assert child.context.trace_id == root_span.context.trace_id
    assert child.status.status_code == trace.StatusCode.ERROR and child.events[0].name == "exception"
    assert child.attributes["status"] == ("cancelled" if isinstance(failure, asyncio.CancelledError) else "failed")
    roles = [s for s in spans if s.name == "mx.refit.orchestrator"]
    assert len(roles) == int(phase != "wait_update_lock")
    if roles:
        assert roles[0].status.status_code == trace.StatusCode.ERROR
        assert child.parent.span_id == roles[0].context.span_id
    else:
        assert child.parent.span_id == root_span.context.span_id
    assert receiver.current is None and receiver._phases.current.get() is None
    assert not trace.get_current_span().get_span_context().is_valid


@pytest.mark.parametrize("invalid", ["json", "baggage"])
def test_invalid_metadata_falls_back_and_startup_adopts_late_offer(refit_trace, invalid):
    receiver, root, carrier, step_dir, exporter, warnings = refit_trace
    marker = step_dir / ".sender_ready"
    if invalid == "json":
        marker.write_text("broken json")
    else:
        publish_offer(marker, 3, {**carrier, "baggage": "step=not-an-integer"})
    with receiver.trace_update(3):
        receiver.trace_accept()
        assert receiver.current.role is None
        with receiver.trace_phase("receiver_ack"):
            pass
        receiver.current.carrier = carrier
        receiver.trace_accept()
    assert warnings
    marker.unlink()
    with receiver.trace_update(3):
        receiver.trace_accept()
        with receiver.trace_phase("confirm_offer"):
            publish_offer(marker, 3, carrier)
        receiver.trace_accept()
        assert receiver.current.role is not None
    root.finish()
    assert len([s for s in exporter.get_finished_spans() if s.name == "mx.refit.orchestrator"]) == 2


def test_disabled_receiver_does_not_read_metadata_or_emit(refit_trace, monkeypatch):
    receiver, root, carrier, step_dir, exporter, warnings = refit_trace
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    (step_dir / ".sender_ready").write_text("broken json")
    with receiver.trace_update(3):
        receiver.trace_accept()
        with receiver.trace_phase("receiver_ack"):
            pass
        assert receiver.current.role is None
    assert not warnings and not exporter.get_finished_spans()


def test_failure_without_legacy_carrier_reports_missing_context(refit_trace):
    receiver, root, carrier, step_dir, exporter, warnings = refit_trace
    (step_dir / ".sender_ready").touch()
    with pytest.raises(TimeoutError), receiver.trace_update(3):
        receiver.trace_accept()
        with receiver.trace_phase("wait_version_marker"):
            raise TimeoutError("version marker missing")
    assert any("no trace context" in warning for warning in warnings)
    assert not exporter.get_finished_spans()
    assert receiver.current is None and receiver._phases.current.get() is None


@pytest.mark.asyncio
async def test_refit_lock_preserves_cancellation_and_parent_lock(refit_trace):
    receiver, root, carrier, step_dir, exporter, _ = refit_trace
    publish_offer(step_dir / ".sender_ready", 3, carrier)
    lock = RefitLock(receiver, "wait_update_lock")
    await lock.acquire()

    async def update():
        with receiver.trace_update(3):
            await lock.acquire()

    task = asyncio.create_task(update())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lock.locked()
    lock.release()
    phase = next(s for s in exporter.get_finished_spans() if s.name == "mx.refit.wait_update_lock")
    assert phase.attributes["status"] == "cancelled"
    assert phase.status.status_code == trace.StatusCode.ERROR


@pytest.mark.asyncio
async def test_rollout_tasks_do_not_export_refit_lock_spans(refit_trace):
    receiver, root, carrier, step_dir, exporter, _ = refit_trace
    publish_offer(step_dir / ".sender_ready", 3, carrier)
    lock = RefitLock(receiver, "wait_scheduling_lock")
    delayed = asyncio.Event()

    async def rollout(wait=False):
        if wait:
            await delayed.wait()
        async with lock:
            assert not trace.get_current_span().get_span_context().is_valid

    with receiver.trace_update(3):
        receiver.trace_accept()
        async with lock:
            pass
        await asyncio.create_task(rollout())
        after_update = asyncio.create_task(rollout(wait=True))
    delayed.set()
    await after_update
    assert len([s for s in exporter.get_finished_spans() if s.name == "mx.refit.wait_scheduling_lock"]) == 1


@pytest.mark.parametrize("failure", [None, RuntimeError("cancellation failed"), asyncio.CancelledError()])
@pytest.mark.asyncio
async def test_cancellation_aggregate_counts_success_and_partial_failure(refit_trace, failure):
    receiver, root, carrier, step_dir, exporter, _ = refit_trace
    publish_offer(step_dir / ".sender_ready", 3, carrier)

    async def drop_group(group_id, *, reason):
        assert reason == "stale"
        if group_id == "second" and failure is not None:
            raise failure
        return 3

    def scope():
        return pytest.raises(type(failure)) if failure is not None else ExitStack()

    with scope(), receiver.trace_update(3), ExitStack() as stack:
        receiver.trace_accept()
        cancellation = CancellationTrace(stack, receiver.trace_phase, minimum=9)
        await cancellation.record_group(drop_group, "first", "stale")
        await cancellation.record_group(drop_group, "second", "stale")
    phase = next(s for s in exporter.get_finished_spans() if s.name == "mx.refit.cancel_stale_rollouts")
    assert phase.attributes["rollout.min_version"] == 9
    assert phase.attributes["rollout.attempted_groups"] == 2
    assert phase.attributes["rollout.cancelled_episodes"] == (3 if failure is not None else 6)
    assert (phase.status.status_code == trace.StatusCode.ERROR) == (failure is not None)
