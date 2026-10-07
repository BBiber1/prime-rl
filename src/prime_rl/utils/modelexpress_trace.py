"""ModelExpress receiver trace lifetime, independent of training and inference runtimes."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from modelexpress import telemetry

from prime_rl.utils.weight_trace import Phase, PhaseRecorder, clock_identity, discovery_phases, read_offer


@dataclass
class ReceiverTrace:
    step: int
    offer: dict = field(default_factory=dict)
    carrier: dict = field(default_factory=dict)
    uid: str = ""
    accepted: bool = False
    role: telemetry.RefitCycle | None = None


class ReceiverTracing:
    def __init__(self, step_dir, staging_mode: str, logger):
        self.step_dir = step_dir
        self.staging_mode = staging_mode
        self.logger = logger
        self._phases = PhaseRecorder(telemetry.enabled)
        self._trace: ContextVar[ReceiverTrace | None] = ContextVar("mx_receiver_update", default=None)

    def trace_phase(self, name: str, attributes: dict | None = None):
        return self._phases.phase(name, attributes)

    def _read_offer(self, state: ReceiverTrace) -> None:
        if state.carrier or not telemetry.enabled():
            return
        try:
            offer = read_offer(self.step_dir(state.step) / ".sender_ready", state.step)
            if offer is not None:
                with telemetry.extracted(offer["trace_context"]):
                    pass
        except (ValueError, OSError) as error:
            self.logger.warning(f"Ignoring sender trace metadata for step {state.step}: {error}")
            return
        if offer is not None:
            state.offer, state.carrier = offer, offer["trace_context"]

    def attributes(self, state: ReceiverTrace) -> dict:
        return {
            "step": state.step,
            "refit.step": state.step,
            "version_uid": state.uid,
            "refit.id": state.uid,
            "refit.phase": "cold" if state.step == 0 else "warm",
            "experiment": os.environ.get("MX_REFIT_EXPERIMENT", ""),
            "staging_mode": self.staging_mode,
            "refit.aggregate": True,
            "role": "orchestrator",
            "rank": 0,
        }

    def _emit_phase(self, state: ReceiverTrace, phase: Phase) -> None:
        parent = state.role.active() if state.role is not None else telemetry.extracted(state.carrier)
        with parent, telemetry.refit_attributes(self.attributes(state)):
            telemetry.completed_span(
                f"mx.refit.{phase.name}", phase.start, phase.end, phase.attributes, error=phase.error
            )

    def trace_accept(self) -> None:
        state = self._trace.get()
        assert state is not None
        state.accepted = True
        self._read_offer(state)
        if not telemetry.enabled() or not state.carrier or state.role is not None:
            return
        state.role = telemetry.RefitCycle(self.attributes(state), name="mx.refit.orchestrator", parent=state.carrier)
        for phase in discovery_phases(state.offer, self._phases.poll, clock_identity()):
            self._emit_phase(state, phase)
        self._phases.bind(lambda phase: self._emit_phase(state, phase))

    @contextmanager
    def trace_update(self, step: int):
        telemetry.configure("prime-rl-orchestrator")
        state = ReceiverTrace(step)
        token = self._trace.set(state)
        error = None
        try:
            with self._phases.session():
                self._read_offer(state)
                try:
                    yield
                    self._read_offer(state)
                    if state.carrier and state.role is None:
                        self._phases.bind(lambda phase: self._emit_phase(state, phase))
                except BaseException as failure:
                    error = failure
                    self._read_offer(state)
                    if state.carrier and state.role is None:
                        if state.accepted:
                            self.trace_accept()
                        else:
                            self._phases.bind(lambda phase: self._emit_phase(state, phase))
                    elif not state.carrier and telemetry.enabled():
                        self.logger.warning(f"Cannot correlate failed receiver update step {step}: no trace context")
                    raise
        finally:
            if state.role is not None:
                state.role.finish(error)
            if state.accepted:
                self._phases.poll.clear()
            self._trace.reset(token)

    @property
    def current(self) -> ReceiverTrace | None:
        session = self._phases.current.get()
        return self._trace.get() if session is not None and session.active else None
