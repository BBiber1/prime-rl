import asyncio
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("modelexpress")
pytest.importorskip("opentelemetry.sdk")

from modelexpress import telemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from prime_rl.utils.modelexpress_trace import ReceiverTracing
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
async def test_receiver_envelope_and_confirmation_do_not_leak_context(refit_trace, legacy):
    receiver, root, carrier, step_dir, exporter, _ = refit_trace
    marker = step_dir / ".sender_ready"
    if legacy:
        marker.touch()
    else:
        publish_offer(marker, 3, carrier)
    with receiver.trace_phase("receiver_scan", {"step": 3}):
        pass
    with receiver.trace_update(3), receiver.trace_phase("confirm_offer"):
        assert marker.exists()
    with receiver.trace_update(3):
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
    assert receiver.current is None and receiver._phases.current.get() is None

    async def rollout():
        assert not trace.get_current_span().get_span_context().is_valid
        assert not telemetry._refit_attributes.get()

    await asyncio.create_task(rollout())
    root.finish()
    spans = exporter.get_finished_spans()
    roles = [s for s in spans if s.name == "mx.refit.orchestrator"]
    assert len(roles) == 1
    role = roles[0]
    children = [s for s in spans if s.parent is not None and s.parent.span_id == role.context.span_id]
    cycle = next(s for s in spans if s.name == "mx.refit.cycle")
    assert role.parent.span_id == cycle.context.span_id
    assert all(s.context.trace_id == role.context.trace_id for s in children)
    assert (role.start_time, role.end_time) == (min(s.start_time for s in children), max(s.end_time for s in children))
    assert any(s.name == "mx.refit.receiver_ack" for s in children)
    confirmation = [s for s in spans if s.name == "mx.refit.confirm_offer"]
    assert len(confirmation) == int(not legacy)
    if confirmation:
        assert confirmation[0].parent.span_id == cycle.context.span_id


@pytest.mark.parametrize(
    "phase",
    [
        "confirm_offer",
        "receiver_ack",
        "wait_version_marker",
        "version_context_lookup",
    ],
)
@pytest.mark.parametrize("failure", [RuntimeError("phase failed"), asyncio.CancelledError()])
def test_early_failure_is_correlated_and_cleans_up(refit_trace, phase, failure):
    receiver, root, carrier, step_dir, exporter, _ = refit_trace
    publish_offer(step_dir / ".sender_ready", 3, carrier)
    with pytest.raises(type(failure)), receiver.trace_update(3):
        if phase != "confirm_offer":
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
    assert len(roles) == int(phase != "confirm_offer")
    if roles:
        assert roles[0].status.status_code == trace.StatusCode.ERROR
        assert child.parent.span_id == roles[0].context.span_id
    else:
        assert child.parent.span_id == root_span.context.span_id
    assert receiver.current is None and receiver._phases.current.get() is None
    assert not trace.get_current_span().get_span_context().is_valid


@pytest.mark.parametrize("invalid", ["json", "baggage"])
def test_invalid_metadata_falls_back_and_confirmation_adopts_late_offer(refit_trace, invalid):
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
        with receiver.trace_phase("confirm_offer"):
            publish_offer(marker, 3, carrier)
    root.finish()
    spans = exporter.get_finished_spans()
    assert len([s for s in spans if s.name == "mx.refit.orchestrator"]) == 1
    confirmation = next(s for s in spans if s.name == "mx.refit.confirm_offer")
    assert confirmation.parent.span_id == next(s for s in spans if s.name == "mx.refit.cycle").context.span_id


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
