"""ModelExpress marker metadata and receiver phase timings."""

import json
import os
import re
import socket
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import monotonic_ns, time_ns

from modelexpress import telemetry


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
    start: int
    end: int
    monotonic_end: int
    attributes: dict
    error: BaseException | None = None


@contextmanager
def record_phase(phases: dict[str, Phase], name: str, attributes: dict | None = None):
    attributes = dict(attributes or {})
    if not telemetry.enabled():
        yield attributes
        return
    start = time_ns()
    error = None
    try:
        yield attributes
    except BaseException as failure:
        error = failure
        raise
    finally:
        phases[name] = Phase(start, time_ns(), monotonic_ns(), attributes, error)


def discovery_phase(offer: dict, scan: Phase, identity: tuple[str, str]) -> Phase | None:
    host, clock = identity
    offered = offer.get("offered_at_monotonic_ns")
    if (
        not clock
        or offer.get("sender_host") != host
        or offer.get("clock_id") != clock
        or scan.attributes.get("step") != offer.get("step")
        or type(offered) is not int
        or offered > scan.monotonic_end
    ):
        return None
    return Phase(scan.end - (scan.monotonic_end - offered), scan.end, scan.monotonic_end, {"offer.clock_shared": True})


class ReceiverTrace:
    def __init__(self, step: int, step_dir: Path, staging_mode: str, logger, samples: dict[str, Phase]):
        self.step, self.step_dir, self.staging_mode, self.logger = step, step_dir, staging_mode, logger
        self.phases = {name: phase for name, phase in samples.items() if phase.attributes.get("step") == step}
        self.offer, self.carrier, self.uid = {}, {}, ""
        self.role = None
        self.metadata_read = False

    def __enter__(self):
        return self

    def __exit__(self, _type, error, _traceback):
        if error is not None and self.role is None:
            self.attach(self.carrier, self.uid)
            if telemetry.enabled() and not self.carrier:
                self.logger.warning(f"Cannot correlate failed receiver update step {self.step}: no trace context")
        if self.role is not None:
            try:
                self.role.finish(error)
            except Exception as failure:
                self.logger.warning(f"Cannot finish receiver telemetry for step {self.step}: {failure}")

    @property
    def attributes(self) -> dict:
        return {
            "step": self.step,
            "refit.step": self.step,
            "version_uid": self.uid,
            "refit.id": self.uid,
            "refit.phase": "cold" if self.step == 0 else "warm",
            "experiment": os.environ.get("MX_REFIT_EXPERIMENT", ""),
            "staging_mode": self.staging_mode,
            "refit.aggregate": True,
            "role": "orchestrator",
            "rank": 0,
        }

    def active(self):
        return self.role.active() if self.role is not None else nullcontext()

    def attach(self, carrier: dict, uid: str) -> None:
        self.carrier, self.uid = carrier, uid
        if not telemetry.enabled() or self.role is not None:
            return
        if not self.metadata_read:
            self.metadata_read = True
            try:
                offer = read_offer(self.step_dir / ".sender_ready", self.step)
                if offer is not None:
                    if carrier and offer["trace_context"].get("traceparent") != carrier.get("traceparent"):
                        raise ValueError("sender offer/version trace mismatch")
                    with telemetry.extracted(offer["trace_context"]):
                        pass
                    self.offer = offer
                    self.carrier = carrier or offer["trace_context"]
            except Exception as error:
                self.logger.warning(f"Ignoring sender trace metadata for step {self.step}: {error}")
        if not self.carrier:
            return
        try:
            telemetry.configure("prime-rl-orchestrator")
            self.role = telemetry.RefitCycle(self.attributes, name="mx.refit.orchestrator", parent=self.carrier)
            scan = self.phases.get("receiver_scan")
            if scan is not None:
                scan.attributes.update(
                    {
                        f"offer.{key}": value
                        for key, value in self.offer.items()
                        if key in {"sender_host", "offered_at_unix_ns", "offered_at_monotonic_ns"}
                    }
                )
                discovery = discovery_phase(self.offer, scan, clock_identity())
                scan.attributes["offer.clock_shared"] = discovery is not None
                if discovery is not None:
                    self._emit("offer_to_discovery", discovery)
            for name, phase in self.phases.items():
                self._emit(name, phase)
        except Exception as error:
            self.logger.warning(f"Cannot start receiver telemetry for step {self.step}: {error}")

    def _emit(self, name: str, phase: Phase) -> None:
        try:
            # Discovery precedes wait_published and belongs directly to the cycle.
            parent = (
                telemetry.extracted(self.carrier) if name in {"receiver_scan", "offer_to_discovery"} else self.active()
            )
            with parent, telemetry.refit_attributes(self.attributes):
                telemetry.completed_span(
                    f"mx.refit.{name}", phase.start, phase.end, phase.attributes, error=phase.error
                )
        except Exception as error:
            self.logger.warning(f"Cannot record receiver phase {name}: {error}")
