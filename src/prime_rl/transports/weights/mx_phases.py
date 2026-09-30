"""Structured phase timing for ModelExpress refits."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from modelexpress import telemetry

RECORD = "mx-refit-phases-v1"


class PhaseTimer:
    """Accumulates named phase durations for one refit and emits them once."""

    def __init__(self, role: str, step: int, version_uid: str, rank: int = 0) -> None:
        self.role = role
        self.rank = rank
        self.step = step
        self.version_uid = version_uid
        self.phases: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self.spans: list[dict] = []
        self.span_gaps: list[dict] = []
        self.hostname = socket.gethostname()
        self.started = time.perf_counter()
        self.elapsed: float | None = None
        self.status = "running"
        self.trace_span = None

    @property
    def trace_attributes(self) -> dict[str, str | int]:
        attributes: dict[str, str | int] = {
            "role": "control" if self.role == "orchestrator" else self.role,
            "rank": self.rank,
            "step": self.step,
            "version_uid": self.version_uid,
        }
        for attribute, variable in (("experiment", "MX_REFIT_EXPERIMENT"), ("staging_mode", "MX_REFIT_STAGING_MODE")):
            if value := os.environ.get(variable):
                attributes[attribute] = value
        return attributes

    @property
    def metric_attributes(self) -> dict[str, str | int]:
        return {key: value for key, value in self.trace_attributes.items() if key not in ("step", "version_uid")}

    def identify(self, version_uid: str) -> None:
        """Set the version ID after discovery."""
        self.version_uid = version_uid
        if self.trace_span is not None and self.trace_span.is_recording():
            self.trace_span.set_attribute("version_uid", version_uid)

    def mark(self, name: str, value: float) -> None:
        """Record a value that is not part of accounted phase time."""
        self.marks[name] = value
        if self.trace_span is not None and self.trace_span.is_recording():
            self.trace_span.set_attribute(f"mx.mark.{name}", value)

    @contextmanager
    def phase(self, name: str, *, timeline: bool = False) -> Iterator[None]:
        started = time.perf_counter()
        first_span = len(self.spans)
        completed = False
        with telemetry.span(f"mx.refit.{name}", self.trace_attributes):
            try:
                yield
                completed = True
            finally:
                ended = time.perf_counter()
                elapsed = ended - started
                self.phases[name] = self.phases.get(name, 0.0) + elapsed
                telemetry.duration(
                    "mx_refit_phase_duration_seconds",
                    elapsed,
                    {**self.metric_attributes, "phase": name, "status": "complete" if completed else "failed"},
                )
                if timeline:
                    children = self.spans[first_span:]
                    parent = self._record_span(name, started, ended, completed)
                    self._record_gaps(parent, children)

    @contextmanager
    def child(self, name: str) -> Iterator[None]:
        """Time a nested operation without counting it again in accounted time."""
        started = time.perf_counter()
        with telemetry.span(f"mx.refit.{name}", self.trace_attributes):
            try:
                yield
            finally:
                self.marks[name] = self.marks.get(name, 0.0) + (time.perf_counter() - started)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Record a child interval, including failed/cancelled calls, on this timer's clock."""
        started = time.perf_counter()
        completed = False
        with telemetry.span(f"mx.refit.{name}", self.trace_attributes):
            try:
                yield
                completed = True
            finally:
                ended = time.perf_counter()
                self.marks[f"{name}_s"] = self.marks.get(f"{name}_s", 0.0) + ended - started
                self._record_span(name, started, ended, completed)

    def _record_span(self, name: str, started: float, ended: float, completed: bool) -> dict:
        span = {
            "name": name,
            "start_offset_s": started - self.started,
            "end_offset_s": ended - self.started,
            "duration_s": ended - started,
            "status": "complete" if completed else "failed",
        }
        self.spans.append(span)
        return span

    def _record_gaps(self, parent: dict, children: list[dict]) -> None:
        """Keep the complement of the child interval union, never their summed durations."""
        cursor = parent["start_offset_s"]
        end = parent["end_offset_s"]
        gaps = []
        for child in sorted(children, key=lambda span: span["start_offset_s"]):
            start = max(parent["start_offset_s"], min(end, child["start_offset_s"]))
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, min(end, child["end_offset_s"]))
        if cursor < end:
            gaps.append((cursor, end))
        for start, stop in gaps:
            self.span_gaps.append(
                {
                    "parent": parent["name"],
                    "start_offset_s": start,
                    "end_offset_s": stop,
                    "duration_s": stop - start,
                }
            )
        self.marks[f"{parent['name']}_uncovered_s"] = self.marks.get(f"{parent['name']}_uncovered_s", 0.0) + sum(
            stop - start for start, stop in gaps
        )

    def payload(self) -> dict:
        """Build the structured timing record."""
        phases = {name: round(value, 6) for name, value in self.phases.items()}
        payload = {
            "record": RECORD,
            "role": self.role,
            "step": self.step,
            "version_uid": self.version_uid,
            "status": self.status,
            "hostname": self.hostname,
            "elapsed_s": round(self.elapsed if self.elapsed is not None else time.perf_counter() - self.started, 6),
            "phases_s": phases,
            "accounted_s": round(sum(phases.values()), 6),
        }
        if self.marks:
            payload["marks"] = {name: round(value, 6) for name, value in self.marks.items()}
        if self.spans:
            payload["span_clock"] = "time.perf_counter offsets from this refit's start on hostname"
            for name, entries in (("spans", self.spans), ("span_gaps", self.span_gaps)):
                payload[name] = [
                    {key: round(value, 6) if isinstance(value, float) else value for key, value in entry.items()}
                    for entry in entries
                ]
        return payload

    def emit(self) -> None:
        line = json.dumps(self.payload())
        if os.environ.get("MX_REFIT_TIMING_STDOUT") == "1":
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            return
        try:
            from prime_rl.utils.logger import get_logger
        except ImportError:
            print(line, flush=True)
        else:
            get_logger().info(line)


@contextmanager
def timed_refit(
    role: str, step: int, version_uid: str, *, rank: int = 0, parent: dict[str, str] | None = None
) -> Iterator[PhaseTimer]:
    """Time one refit cycle for ``role`` and emit the split on the way out."""
    telemetry.configure(f"prime-rl-{role}")
    timer = PhaseTimer(role, step, version_uid, rank)
    with telemetry.extracted(parent or {}):
        with telemetry.span("mx.refit", timer.trace_attributes) as trace_span:
            timer.trace_span = trace_span
            with telemetry.refit_attributes(timer.trace_attributes):
                try:
                    yield timer
                except BaseException:
                    timer.status = "failed"
                    raise
                else:
                    timer.status = "complete"
                finally:
                    timer.elapsed = time.perf_counter() - timer.started
                    if trace_span.is_recording():
                        trace_span.set_attribute("status", timer.status)
                    telemetry.duration(
                        "mx_refit_total_duration_seconds",
                        timer.elapsed,
                        {**timer.metric_attributes, "status": timer.status},
                    )
                    timer.emit()
