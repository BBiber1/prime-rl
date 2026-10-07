import asyncio
import json

import pytest

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
