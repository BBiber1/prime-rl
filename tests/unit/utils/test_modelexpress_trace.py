import asyncio
import json
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

from prime_rl.utils import modelexpress_trace
from prime_rl.utils.modelexpress_trace import (
    Phase,
    ReceiverTrace,
    discovery_phase,
    publish_offer,
    read_offer,
    record_phase,
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


def test_discovery_latency_requires_shared_clock_and_matching_step():
    scan = Phase(200, 220, 130, {"step": 3})
    offer = {"step": 3, "sender_host": "host", "clock_id": "boot/ns", "offered_at_monotonic_ns": 60}
    elapsed = discovery_phase(offer, scan, ("host", "boot/ns"))
    assert (elapsed.start, elapsed.end) == (150, 220)
    assert (scan.start, scan.end) == (200, 220)
    for identity in (("other-host", "boot/ns"), ("host", "other-ns"), ("host", "")):
        assert discovery_phase(offer, scan, identity) is None
    assert discovery_phase({**offer, "offered_at_monotonic_ns": 140}, scan, ("host", "boot/ns")) is None
    assert discovery_phase({**offer, "step": 4}, scan, ("host", "boot/ns")) is None


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
    logger = SimpleNamespace(warning=warnings.append)
    yield root, carrier, step_dir, exporter, logger, warnings
    root.finish()
    provider.shutdown()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.asyncio
async def test_receiver_envelope_and_confirmation_do_not_leak_context(refit_trace, legacy, monkeypatch):
    root, carrier, step_dir, exporter, logger, _ = refit_trace
    marker = step_dir / ".sender_ready"
    if legacy:
        marker.touch()
    else:
        publish_offer(marker, 3, carrier)
    samples = {}
    with record_phase(samples, "receiver_scan", {"step": 3}):
        pass
    metadata_reads = []
    lookup_done = []
    original_read = modelexpress_trace.read_offer

    def read_after_lookup(marker, step):
        assert (step_dir / ".receiver_ready").exists() and lookup_done
        metadata_reads.append(step)
        return original_read(marker, step)

    monkeypatch.setattr(modelexpress_trace, "read_offer", read_after_lookup)
    with record_phase(samples, "confirm_offer", {"step": 3}):
        assert marker.exists()
    with ReceiverTrace(3, step_dir, "H2D", logger, samples) as receiver:
        assert receiver.role is None
        with record_phase(receiver.phases, "receiver_ack"):
            (step_dir / ".receiver_ready").touch()
        with record_phase(receiver.phases, "wait_version_marker"):
            pass
        with record_phase(receiver.phases, "version_context_lookup"):
            lookup_done.append(True)
        assert receiver.role is None and not metadata_reads and not exporter.get_finished_spans()
        receiver.attach(carrier, "v3")
        receiver.attach(carrier, "v3")
    assert metadata_reads == [3]

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
    confirmation = next(s for s in spans if s.name == "mx.refit.confirm_offer")
    assert confirmation.parent.span_id == role.context.span_id


@pytest.mark.parametrize("phase", ["receiver_ack", "wait_version_marker", "version_context_lookup"])
@pytest.mark.parametrize("failure", [RuntimeError("phase failed"), asyncio.CancelledError()])
def test_early_failure_is_correlated_and_cleans_up(refit_trace, phase, failure):
    root, carrier, step_dir, exporter, logger, _ = refit_trace
    publish_offer(step_dir / ".sender_ready", 3, carrier)
    with pytest.raises(type(failure)) as raised, ReceiverTrace(3, step_dir, "H2D", logger, {}) as receiver:
        with record_phase(receiver.phases, phase):
            raise failure
    assert raised.value is failure and receiver.phases[phase].error is failure
    root.finish()
    spans = exporter.get_finished_spans()
    child = next(s for s in spans if s.name == f"mx.refit.{phase}")
    root_span = next(s for s in spans if s.name == "mx.refit.cycle")
    assert child.context.trace_id == root_span.context.trace_id
    assert child.status.status_code == trace.StatusCode.ERROR and child.events[0].name == "exception"
    assert child.attributes["status"] == ("cancelled" if isinstance(failure, asyncio.CancelledError) else "failed")
    role = next(s for s in spans if s.name == "mx.refit.orchestrator")
    assert role.status.status_code == trace.StatusCode.ERROR
    assert child.parent.span_id == role.context.span_id
    assert not trace.get_current_span().get_span_context().is_valid


@pytest.mark.parametrize("invalid", ["json", "baggage", "trace"])
def test_invalid_metadata_falls_back_to_version_carrier(refit_trace, invalid):
    root, carrier, step_dir, exporter, logger, warnings = refit_trace
    marker = step_dir / ".sender_ready"
    if invalid == "json":
        marker.write_text("broken json")
    elif invalid == "baggage":
        publish_offer(marker, 3, {**carrier, "baggage": "step=not-an-integer"})
    else:
        publish_offer(marker, 3, CARRIER)
    with ReceiverTrace(3, step_dir, "H2D", logger, {}) as receiver:
        with record_phase(receiver.phases, "receiver_ack"):
            pass
        receiver.attach(carrier, "v3")
        assert not receiver.offer and receiver.role is not None
    assert warnings
    root.finish()
    spans = exporter.get_finished_spans()
    role = next(s for s in spans if s.name == "mx.refit.orchestrator")
    cycle = next(s for s in spans if s.name == "mx.refit.cycle")
    assert role.context.trace_id == cycle.context.trace_id


def test_disabled_receiver_does_not_read_metadata_or_emit(refit_trace, monkeypatch):
    root, carrier, step_dir, exporter, logger, warnings = refit_trace
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    (step_dir / ".sender_ready").write_text("broken json")
    with ReceiverTrace(3, step_dir, "H2D", logger, {}) as receiver:
        with record_phase(receiver.phases, "receiver_ack"):
            pass
        receiver.attach(carrier, "v3")
        assert receiver.role is None and not receiver.phases
    assert not warnings and not exporter.get_finished_spans()


def test_failure_without_legacy_carrier_reports_missing_context(refit_trace):
    root, carrier, step_dir, exporter, logger, warnings = refit_trace
    (step_dir / ".sender_ready").touch()
    with pytest.raises(TimeoutError), ReceiverTrace(3, step_dir, "H2D", logger, {}) as receiver:
        with record_phase(receiver.phases, "wait_version_marker"):
            raise TimeoutError("version marker missing")
    assert any("no trace context" in warning for warning in warnings)
    assert not exporter.get_finished_spans()


@pytest.mark.parametrize("target", ["phase", "role_start", "role_finish"])
def test_phase_export_failure_preserves_operation_and_original_error(refit_trace, monkeypatch, target):
    root, carrier, step_dir, exporter, logger, warnings = refit_trace
    publish_offer(step_dir / ".sender_ready", 3, carrier)

    def broken_export(*args, **kwargs):
        raise RuntimeError("export unavailable")

    if target == "phase":
        monkeypatch.setattr(telemetry, "completed_span", broken_export)
    elif target == "role_start":
        monkeypatch.setattr(telemetry, "RefitCycle", broken_export)
    else:
        create_cycle = telemetry.RefitCycle

        def cycle_with_broken_finish(*args, **kwargs):
            role = create_cycle(*args, **kwargs)
            finish = role.finish

            def broken_finish(error=None):
                finish(error)
                raise RuntimeError("export unavailable")

            monkeypatch.setattr(role, "finish", broken_finish)
            return role

        monkeypatch.setattr(telemetry, "RefitCycle", cycle_with_broken_finish)
    completed = []
    with ReceiverTrace(3, step_dir, "H2D", logger, {}) as receiver:
        with record_phase(receiver.phases, "receiver_ack"):
            completed.append("ack")
        receiver.attach(carrier, "v3")
    failure = ValueError("transfer failed")
    with pytest.raises(ValueError) as raised, ReceiverTrace(3, step_dir, "H2D", logger, {}) as receiver:
        with record_phase(receiver.phases, "version_context_lookup"):
            raise failure
    assert raised.value is failure and completed == ["ack"]
    assert warnings and all("export unavailable" in message for message in warnings)
