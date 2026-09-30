"""ModelExpress version naming across restarts."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from prime_rl.configs.trainer import MXRefitWeightBroadcastConfig
from prime_rl.transports.weights.base import SENDER_READY_MARKER
from prime_rl.transports.weights.mx_refit import (
    MXRefitWeightReceiver,
    MXRefitWeightSender,
    weight_version_uid,
)
from prime_rl.utils.pathing import get_broadcast_dir

RUN_UID = "testrun"


def make_sender(output_dir: Path, timeout: int = 5, handshake_mode: str = "object") -> MXRefitWeightSender:
    config = MXRefitWeightBroadcastConfig(run_uid=RUN_UID, timeout=timeout, handshake_mode=handshake_mode)
    return MXRefitWeightSender(output_dir, config, parallel_dims=None, model_name="model")


def make_receiver(output_dir: Path, timeout: int = 5) -> MXRefitWeightReceiver:
    config = MXRefitWeightBroadcastConfig(run_uid="unused-by-this-side", timeout=timeout)
    return MXRefitWeightReceiver(get_broadcast_dir(output_dir), config, admin_plane=None, model_name="model")


def offer(sender: MXRefitWeightSender, step: int) -> Path:
    step_dir = sender.step_dir(step)
    step_dir.mkdir(parents=True, exist_ok=True)
    sender._offer(step_dir)
    return step_dir


@pytest.mark.parametrize("mode", ["object", "tensor"])
def test_restart_uses_new_version_id(tmp_path, mode):
    first = make_sender(tmp_path, handshake_mode=mode)
    offer(first, 3)
    restarted = make_sender(tmp_path, handshake_mode=mode)
    offer(restarted, 3)

    assert first._offer_token != restarted._offer_token
    assert weight_version_uid(first._offer_token, 3) != weight_version_uid(restarted._offer_token, 3)


@pytest.mark.parametrize("mode", ["object", "tensor"])
def test_each_offer_uses_new_version_id(tmp_path, mode):
    sender = make_sender(tmp_path, handshake_mode=mode)
    offer(sender, 3)
    first = sender._offer_token
    offer(sender, 3)

    assert first != sender._offer_token


def test_version_id_contains_run_and_step(tmp_path):
    sender = make_sender(tmp_path)
    offer(sender, 7)
    uid = weight_version_uid(sender._offer_token, 7)

    assert uid.startswith(f"{RUN_UID}.")
    assert uid.rpartition(":")[2] == "7"


def test_receiver_reads_offered_version_id(tmp_path):
    sender = make_sender(tmp_path)
    receiver = make_receiver(tmp_path)
    offer(sender, 4)

    token = asyncio.run(receiver._read_offer_token(4))

    assert token == sender._offer_token
    assert weight_version_uid(token, 4) == weight_version_uid(sender._offer_token, 4)


def test_receiver_follows_restarted_trainer(tmp_path):
    receiver = make_receiver(tmp_path)
    stale = make_sender(tmp_path)
    offer(stale, 5)
    stale_uid = weight_version_uid(stale._offer_token, 5)

    restarted = make_sender(tmp_path)
    offer(restarted, 5)

    resolved = weight_version_uid(asyncio.run(receiver._read_offer_token(5)), 5)
    assert resolved == weight_version_uid(restarted._offer_token, 5)
    assert resolved != stale_uid


def test_offer_marker_is_atomic(tmp_path):
    sender = make_sender(tmp_path)
    step_dir = offer(sender, 2)

    assert (step_dir / SENDER_READY_MARKER).read_text().strip() == sender._offer_token
    assert [path.name for path in step_dir.iterdir()] == [SENDER_READY_MARKER]


def test_receiver_run_uid_is_not_used(tmp_path):
    sender = make_sender(tmp_path)
    receiver = make_receiver(tmp_path)
    offer(sender, 8)

    assert receiver.config.run_uid != sender.config.run_uid
    assert asyncio.run(receiver._read_offer_token(8)) == sender._offer_token


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("step", [0, 3])
def test_native_cycle_covers_offer_roles_and_completion(tmp_path, monkeypatch, failed, step):
    pytest.importorskip("opentelemetry.sdk")
    from modelexpress import telemetry
    from modelexpress_rl import WeightVersionState
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import prime_rl.transports.weights.mx_refit as mx_refit
    from prime_rl.transports.weights.mx_phases import timed_refit

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused/v1/traces")
    monkeypatch.setenv("MX_REFIT_EXPERIMENT", "device")
    monkeypatch.setattr(telemetry, "_configured_pid", os.getpid())
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("cycle-test"))
    barriers = []
    monkeypatch.setattr(mx_refit.dist, "barrier", lambda: barriers.append(True))

    class Control:
        def create_weight_version(self, **kwargs):
            pass

        def get_weight_version(self, uid):
            return SimpleNamespace(state=WeightVersionState.RELEASING)

    class Client:
        def publish_version(self, **kwargs):
            if failed:
                raise RuntimeError("publication failed")

        def release_version(self, **kwargs):
            pass

    sender = make_sender(tmp_path)
    sender.world = SimpleNamespace(rank=0, world_size=1, is_master=True)
    sender._control = Control()
    sender._client = Client()
    sender._initialized = True
    step_dir = offer(sender, step)
    carrier = json.loads((step_dir / mx_refit.TRACE_CONTEXT_MARKER).read_text())
    assert sender._cycle.is_recording()
    assert [s.name for s in exporter.get_finished_spans()] == ["mx.refit.offer"]
    uid = weight_version_uid(sender._offer_token, step)
    # All role roots use the cycle carrier, independently of the offer's span.
    for role in ("orchestrator", "generator", "server"):
        with timed_refit(role, step, uid, parent=carrier):
            pass
    if failed:
        with pytest.raises(RuntimeError, match="publication failed"):
            sender._broadcast(None, step, step_dir)
        assert sender._cycle is None
        assert len(barriers) == 1  # No completion collective after a failure.
    else:
        sender._broadcast(None, step, step_dir)
        assert len(barriers) == 3  # Publish, rendezvous, then closed-span completion.
        assert sender._cycle.is_recording()
        sender._clean(step)
        assert sender._cycle is None
    spans = exporter.get_finished_spans()
    cycle = next(s for s in spans if s.name == "mx.refit.cycle")
    assert cycle.parent is None and cycle.attributes["refit.root"] is True
    assert cycle.attributes["experiment"] == "device" and cycle.attributes["step"] == step
    assert cycle.attributes["refit.phase"] == ("cold" if step == 0 else "warm") and cycle.attributes["refit.id"] == uid
    roots = [s for s in spans if s.name in ("mx.refit.offer", "mx.refit")]
    assert len(roots) == 5
    assert all(s.parent.span_id == cycle.context.span_id for s in roots)
    assert all(s.context.trace_id == cycle.context.trace_id for s in spans)
    assert all(cycle.start_time <= s.start_time <= s.end_time <= cycle.end_time for s in spans)
    assert cycle.attributes["status"] == ("failed" if failed else "complete")
    provider.shutdown()
