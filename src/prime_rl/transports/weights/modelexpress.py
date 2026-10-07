"""FSDP weight updates through ModelExpress's public clients."""

import asyncio
import atexit
import os
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from modelexpress import telemetry
from modelexpress_rl import (
    FSDPTrainerContext,
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionState,
)
from torch import nn

from prime_rl.configs.shared import ModelExpressWeightBroadcastConfig
from prime_rl.transports.weights.base import WeightReceiver, WeightSender
from prime_rl.utils.pathing import wait_for_path

VERSION_MARKER = ".mx_version"
INSTALLED_MARKER = ".installed"


class ModelExpressWeightSender(WeightSender):
    def __init__(self, output_dir: Path, config: ModelExpressWeightBroadcastConfig):
        super().__init__(output_dir, config.timeout)
        if config.model_name is None:
            raise ValueError("modelexpress requires model_name in the broadcast config")
        self.config = config
        self.model_name = config.model_name
        self._trainer: ModelExpressTrainerClient | None = None
        self._control: ModelExpressControlClient | None = None
        self._mesh_id: str | None = None
        self._cycle: telemetry.RefitCycle | None = None
        self._trainers: telemetry.RefitCycle | None = None
        self._role: telemetry.RefitCycle | None = None
        self._carrier: dict[str, str] = {}
        self._trainer_carrier: dict[str, str] = {}
        self._uid = ""

    def _attributes(self, step: int, uid: str = "") -> dict:
        return {
            "step": step,
            "version_uid": uid,
            "refit.id": uid,
            "refit.step": step,
            "refit.phase": "cold" if step == 0 else "warm",
            "experiment": os.environ.get("MX_REFIT_EXPERIMENT", ""),
            "staging_mode": self.config.staging_mode,
            "refit.aggregate": True,
        }

    def _start_cycle(self, step: int) -> None:
        telemetry.configure("prime-rl-trainer")
        self._carrier, self._trainer_carrier = {}, {}
        attributes = self._attributes(step)
        self._cycle = telemetry.RefitCycle(
            {
                **attributes,
                "refit.root": True,
                "refit.expected_trainers": self.world.world_size,
                "refit.expected_generators": self.config.inference_world_size,
            }
        )
        self._cycle.inject(self._carrier)
        self._trainers = telemetry.RefitCycle(
            {**attributes, "role": "trainer"}, name="mx.refit.trainers", parent=self._carrier
        )
        self._trainers.inject(self._trainer_carrier)
        self._role = telemetry.RefitCycle(
            {**attributes, "role": "trainer", "rank": self.world.rank},
            name="mx.refit.trainer",
            parent=self._trainer_carrier,
        )

    def _wait_for_receiver_ready(self, step_dir: Path) -> None:
        step = int(step_dir.name.removeprefix("step_"))
        try:
            self._start_cycle(step)
            with (
                self._role.active(),
                telemetry.refit_attributes(self._attributes(step), role="trainer", rank=self.world.rank),
                telemetry.span("mx.refit.wait_receiver_ready", {"wait.marker": str(step_dir / ".receiver_ready")}),
            ):
                super()._wait_for_receiver_ready(step_dir)
        except BaseException as error:
            self._finish(error)
            raise

    def _finish(self, error: BaseException | None = None) -> None:
        for envelope in (self._role, self._trainers, self._cycle):
            if envelope is not None:
                envelope.finish(error)

    def _clean(self, step: int) -> None:
        error = None
        try:
            with (
                self._role.active(),
                telemetry.refit_attributes(self._attributes(step, self._uid), role="trainer", rank=self.world.rank),
                telemetry.span("mx.refit.broadcast_cleanup"),
            ):
                super()._clean(step)
        except BaseException as failure:
            error = failure
            raise
        finally:
            self._finish(error)

    def _initialize(self, model: nn.Module) -> None:
        tensors = model.state_dict()
        keep_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
        overrides = {name: torch.float32 for name in tensors if keep_fp32 is not None and keep_fp32(name)}
        self._trainer = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                engine_context=FSDPTrainerContext(wire_dtype_overrides=overrides),
                model_name=self.model_name,
                device_id=self.world.local_rank,
                server_url=f"{self.config.host}:{self.config.port}",
                staging_mode=TrainerStagingMode[self.config.staging_mode],
                payload_format=WeightPayloadFormat.FULL_TENSOR,
            )
        )
        atexit.register(self._trainer.close)
        binding = self._trainer.bind_tensors(tensors)
        workers = [None] * self.world.world_size
        dist.all_gather_object(workers, (self._trainer.worker_id, binding))
        if self.world.is_master:
            self._control = ModelExpressControlClient.connect(server_url=f"{self.config.host}:{self.config.port}")
            atexit.register(self._control.close)
            mesh = self._control.create_trainer_mesh(
                model_name=self.model_name,
                idempotency_key=uuid.uuid4().hex,
                workers=dict(workers),
            )
            self._mesh_id = mesh.mesh_id
            atexit.register(self._control.delete_trainer_mesh, self._mesh_id)

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        try:
            initialized = None
            if self.world.is_master:
                with self._role.active(), telemetry.refit_attributes(self._attributes(step), role="trainer", rank=0):
                    if self._trainer is None:
                        with telemetry.span("mx.refit.trainer_initialize"):
                            self._initialize(model)
                    version = self._offer_version(step, step_dir)
            else:
                telemetry.configure("prime-rl-trainer")
                version = None
                if self._trainer is None:
                    start = time.time_ns()
                    with telemetry.untraced():
                        self._initialize(model)
                    initialized = (start, time.time_ns())
            broadcast_start = time.time_ns()
            offered = [version, self._carrier, self._trainer_carrier]
            dist.broadcast_object_list(offered, src=0)
            self._uid, self._carrier, self._trainer_carrier = offered
            broadcast_end = time.time_ns()
            if not self.world.is_master:
                self._role = telemetry.RefitCycle(
                    {**self._attributes(step, self._uid), "role": "trainer", "rank": self.world.rank},
                    name="mx.refit.trainer",
                    parent=self._trainer_carrier,
                )
            attributes = {"version_uid": self._uid, "refit.id": self._uid}
            self._role.set_attributes(attributes)
            with (
                self._role.active(),
                telemetry.refit_attributes(self._attributes(step, self._uid), role="trainer", rank=self.world.rank),
            ):
                if initialized is not None:
                    telemetry.completed_span("mx.refit.trainer_initialize", *initialized)
                telemetry.completed_span("mx.refit.version_broadcast", broadcast_start, broadcast_end)
                self._publish_and_wait(step, step_dir)
            if not self.world.is_master:
                self._role.finish()
        except BaseException as error:
            self._finish(error)
            raise

    def _offer_version(self, step: int, step_dir: Path) -> str | None:
        if self.world.is_master:
            assert self._control is not None and self._mesh_id is not None
            version = self._control.create_weight_version(
                model_name=self.model_name,
                idempotency_key=uuid.uuid4().hex,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                trainer_mesh_id=self._mesh_id,
                version_number=step,
                trace_context=self._carrier,
            )
            uid = version.version_id
            attributes = {"version_uid": uid, "refit.id": uid}
            self._cycle.set_attributes(attributes)
            self._trainers.set_attributes(attributes)
            with telemetry.span("mx.refit.version_marker_publish", attributes):
                marker = step_dir / VERSION_MARKER
                pending = marker.with_suffix(".pending")
                pending.write_text(uid)
                pending.replace(marker)
            return uid
        return None

    def _publish_and_wait(self, step: int, step_dir: Path) -> None:
        assert self._trainer is not None
        version = WeightVersionRef(self._uid)
        attributes = {"version_uid": self._uid, "refit.id": self._uid}
        self._role.set_attributes(attributes)
        with telemetry.refit_attributes(attributes):
            with telemetry.span("mx.refit.publish"):
                self._trainer.publish_version(version=version)
            installation_error: list[Exception | None] = [None]
            if self.world.is_master:
                try:
                    installed = step_dir / INSTALLED_MARKER
                    deadline = time.monotonic() + self.timeout
                    with telemetry.span("mx.refit.wait_installed", {"wait.marker": str(installed)}):
                        while not installed.exists():
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    f"Inference did not install version {self._uid} within {self.timeout}s"
                                )
                            time.sleep(0.1)
                        if installed.read_text() != self._uid:
                            raise RuntimeError("Inference acknowledged a different weight version")
                except Exception as exc:
                    installation_error[0] = exc
            dist.broadcast_object_list(installation_error, src=0)
            if installation_error[0] is not None:
                raise installation_error[0]
            with telemetry.span("mx.refit.release"):
                self._trainer.release_version(version=version)
            with telemetry.span("mx.refit.trainer_barrier"):
                dist.barrier()


class ModelExpressWeightReceiver(WeightReceiver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._published_wait: tuple[int, int, int] | None = None

    async def wait_published(self, step: int, cancelled=None) -> None:
        self._published_wait = None
        if not telemetry.enabled():
            await super().wait_published(step, cancelled=cancelled)
            return
        start = time.time_ns()
        await super().wait_published(step, cancelled=cancelled)
        self._published_wait = (step, start, time.time_ns())

    async def initialize(self) -> None:
        self._control = ModelExpressControlClient.connect(server_url=f"{self.config.host}:{self.config.port}")
        atexit.register(self._control.close)
        await self.admin_plane.initialize_modelexpress(
            self.config.host,
            self.config.port,
            self.config.timeout,
            self.config.inference_world_size,
            self.config.staging_buffer_bytes,
            self.config.staging_buffers_count,
        )

    async def receive(self, step: int) -> None:
        published_wait, self._published_wait = self._published_wait, None
        telemetry.configure("prime-rl-orchestrator")
        attributes = {
            "step": step,
            "refit.step": step,
            "refit.phase": "cold" if step == 0 else "warm",
            "experiment": os.environ.get("MX_REFIT_EXPERIMENT", ""),
            "staging_mode": self.config.staging_mode,
            "refit.aggregate": True,
        }
        ack_start = time.time_ns()
        self._ack(step)
        ack_end = time.time_ns()
        marker = self.step_dir(step) / VERSION_MARKER
        wait_start = time.time_ns()
        await asyncio.wait_for(wait_for_path(marker, interval=0.01), timeout=self.config.timeout)
        uid = marker.read_text()
        wait_end = time.time_ns()
        lookup_start = time.time_ns()
        with telemetry.untraced():
            version = await asyncio.to_thread(self._control.get_weight_version, uid)
        lookup_end = time.time_ns()
        carrier = version.trace_context
        attributes.update({"version_uid": uid, "refit.id": uid})
        with (
            telemetry.extracted(carrier),
            telemetry.refit_attributes(attributes, role="orchestrator", rank=0),
            telemetry.refit_span("mx.refit.orchestrator"),
        ):
            if published_wait is not None and published_wait[0] == step:
                telemetry.completed_span("mx.refit.wait_published", *published_wait[1:])
            telemetry.completed_span("mx.refit.receiver_ack", ack_start, ack_end)
            telemetry.completed_span("mx.refit.wait_version_marker", wait_start, wait_end, {"wait.marker": str(marker)})
            telemetry.completed_span("mx.refit.version_context_lookup", lookup_start, lookup_end)
            await self._receive_version(step, uid, carrier, version)

    async def _receive_version(self, step: int, uid: str, carrier: dict, version) -> None:
        deadline = time.monotonic() + self.config.timeout
        with telemetry.span("mx.refit.wait_version_ready"):
            while True:
                if version.state is WeightVersionState.READY:
                    break
                if version.state is WeightVersionState.RELEASING:
                    raise RuntimeError(f"Weight version {uid} was retired before installation")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Weight version {uid} was not ready within {self.config.timeout}s")
                await asyncio.sleep(0.1)
                version = await asyncio.to_thread(self._control.get_weight_version, uid)
        with telemetry.span("mx.refit.inference_update"):
            await self.admin_plane.update_modelexpress_weights(version_uid=uid, step=step, trace_context=carrier)
        with telemetry.span("mx.refit.retire"):
            await asyncio.to_thread(self._control.delete_weight_version, uid)
        with telemetry.span("mx.refit.installed_marker_publish"):
            installed = self.step_dir(step) / INSTALLED_MARKER
            pending = installed.with_suffix(".pending")
            pending.write_text(uid)
            pending.replace(installed)
