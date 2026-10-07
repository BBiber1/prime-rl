"""Bounded phase timing without installing a tracing context on application tasks."""

import json
import os
import re
import socket
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import monotonic_ns, time_ns
from typing import Any


def run_id() -> str:
    for item in os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "").split(","):
        key, _, value = item.partition("=")
        if key.strip() == "mx.experiment.run_id":
            return value.strip()
    return os.environ.get("MX_REFIT_RUN_ID", "")


@lru_cache(maxsize=1)
def clock_identity() -> tuple[str, str]:
    host = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        namespace = os.readlink("/proc/self/ns/time")
    except OSError:
        return host, ""
    return host, f"{boot}/{namespace}"


def publish_offer(marker: Path, step: int, carrier: dict[str, str]) -> None:
    host, clock = clock_identity()
    payload = {
        "schema_version": 1,
        "step": step,
        "run_id": run_id(),
        "trace_context": carrier,
        "sender_host": host,
        "clock_id": clock,
        "offered_at_unix_ns": time_ns(),
        "offered_at_monotonic_ns": monotonic_ns(),
    }
    pending = marker.with_suffix(".pending")
    pending.write_text(json.dumps(payload))
    pending.replace(marker)


def read_offer(marker: Path, step: int) -> dict | None:
    try:
        raw = marker.read_text()
    except FileNotFoundError:
        return None
    if not raw.strip():
        return None
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        raise ValueError("unsupported sender offer schema")
    if type(payload.get("step")) is not int or payload["step"] != step or payload.get("run_id") != run_id():
        raise ValueError("sender offer step/run mismatch")
    for name in ("sender_host", "clock_id"):
        if not isinstance(payload.get(name), str):
            raise ValueError(f"invalid sender offer {name}")
    for name in ("offered_at_unix_ns", "offered_at_monotonic_ns"):
        if type(payload.get(name)) is not int or payload[name] <= 0:
            raise ValueError(f"invalid sender offer {name}")
    carrier = payload.get("trace_context")
    if not isinstance(carrier, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in carrier.items()):
        raise ValueError("invalid sender trace carrier")
    parent = re.fullmatch(r"00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}", carrier.get("traceparent", ""))
    if parent is None or int(parent[1], 16) == 0 or int(parent[2], 16) == 0:
        raise ValueError("invalid sender traceparent")
    return payload


@dataclass
class Phase:
    name: str
    start: int
    end: int
    monotonic_start: int
    monotonic_end: int
    attributes: dict[str, Any]
    error: BaseException | None = None


@dataclass
class PhaseSession:
    pending: deque[Phase] = field(default_factory=lambda: deque(maxlen=32))
    emit: Callable[[Phase], None] | None = None
    dropped: int = 0


class PhaseRecorder:
    def __init__(self, enabled: Callable[[], bool]):
        self.enabled = enabled
        self.current: ContextVar[PhaseSession | None] = ContextVar("weight_update_phases", default=None)
        self.poll: dict[str, Phase] = {}

    @contextmanager
    def session(self) -> Iterator[PhaseSession]:
        session = PhaseSession()
        token = self.current.set(session)
        try:
            yield session
        finally:
            session.pending.clear()
            self.current.reset(token)

    def bind(self, emit: Callable[[Phase], None]) -> None:
        session = self.current.get()
        assert session is not None
        session.emit = emit
        if session.pending and session.dropped:
            session.pending[0].attributes["trace.dropped_phases"] = session.dropped
        while session.pending:
            emit(session.pending.popleft())

    @contextmanager
    def phase(self, name: str, attributes: dict | None = None) -> Iterator[dict]:
        attrs = dict(attributes or {})
        if not self.enabled():
            yield attrs
            return
        session = self.current.get()
        start, mono_start = time_ns(), monotonic_ns()
        error = None
        try:
            yield attrs
        except BaseException as failure:
            error = failure
            raise
        finally:
            phase = Phase(name, start, time_ns(), mono_start, monotonic_ns(), attrs, error)
            if session is not None:
                if session.emit is not None:
                    session.emit(phase)
                else:
                    if len(session.pending) == session.pending.maxlen:
                        session.dropped += 1
                    session.pending.append(phase)
            elif name in {"watcher_scan", "watcher_poll_sleep"}:
                self.poll[name] = phase


def discovery_phases(offer: dict, poll: dict[str, Phase], local_identity: tuple[str, str]) -> list[Phase]:
    """Clip idle samples only when sender and receiver share a monotonic clock."""
    scan = poll.get("watcher_scan")
    if scan is None:
        return []
    host, clock = local_identity
    shared = bool(clock) and offer.get("sender_host") == host and offer.get("clock_id") == clock
    offered = offer.get("offered_at_monotonic_ns")
    attrs = {
        **scan.attributes,
        "offer.clock_shared": shared,
        **{
            f"offer.{key}": offer[key]
            for key in ("sender_host", "offered_at_unix_ns", "offered_at_monotonic_ns")
            if key in offer
        },
    }
    if not shared or not isinstance(offered, int) or offered > scan.monotonic_end:
        sleep = poll.get("watcher_poll_sleep")
        if sleep is not None:
            attrs["watcher.previous_sleep_s"] = (sleep.monotonic_end - sleep.monotonic_start) / 1e9
        return [Phase(scan.name, scan.start, scan.end, scan.monotonic_start, scan.monotonic_end, attrs, scan.error)]
    phases = []
    for sample in poll.values():
        begin, end = max(offered, sample.monotonic_start), sample.monotonic_end
        if end >= begin:
            phases.append(
                Phase(
                    sample.name,
                    sample.end - (end - begin),
                    sample.end,
                    begin,
                    end,
                    {**sample.attributes, "offer.clock_shared": True},
                    sample.error,
                )
            )
    phases.append(
        Phase(
            "offer_to_discovery",
            scan.end - (scan.monotonic_end - offered),
            scan.end,
            offered,
            scan.monotonic_end,
            attrs,
        )
    )
    return phases
