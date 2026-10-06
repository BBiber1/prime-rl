"""FSDP weight updates through ModelExpress's public clients."""

import asyncio
import atexit
import json
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
    def __init__(self, output_dir: Path, config: ModelExpressWeightBroadcastConfig, model_name: str):
        super().__init__(output_dir, config.timeout)
        self.config = config
        self.model_name = model_name
        self._trainer: ModelExpressTrainerClient | None = None
        self._control: ModelExpressControlClient | None = None
        self._mesh_id: str | None = None
        self._cycle: telemetry.RefitCycle | None = None
        self._carrier: dict[str, str] = {}

    def _start_cycle(self, step: int) -> None:
        telemetry.configure("prime-rl-trainer")
        self._cycle = None
        self._carrier = {}
        if telemetry.enabled():
            self._cycle = telemetry.RefitCycle({**self._attributes(step), "refit.root": True})
            self._cycle.inject(self._carrier)

    def _set_cycle(self, step: int) -> None:
        carrier = [self._carrier]
        broadcast_start = time.time_ns()
        dist.broadcast_object_list(carrier, src=0)
        broadcast_end = time.time_ns()
        self._carrier = carrier[0]
        with telemetry.extracted(self._carrier):
            telemetry.completed_span(
                "mx.refit.trace_context_broadcast", broadcast_start, broadcast_end, self._attributes(step)
            )

    def _wait_for_receiver_ready(self, step_dir: Path) -> None:
        try:
            self._start_cycle(int(step_dir.name.removeprefix("step_")))
            with (
                telemetry.extracted(self._carrier),
                telemetry.span(
                    "mx.refit.wait_receiver_ready",
                    {
                        **self._attributes(int(step_dir.name.removeprefix("step_"))),
                        "wait.marker": str(step_dir / ".receiver_ready"),
                        "wait.poll_interval_s": 0.1,
                    },
                ),
            ):
                super()._wait_for_receiver_ready(step_dir)
        except BaseException as error:
            if self._cycle is not None:
                self._cycle.finish(error)
            raise

    def _attributes(self, step: int, uid: str = "") -> dict:
        return {
            "role": "trainer",
            "rank": self.world.rank,
            "step": step,
            "version_uid": uid,
            "refit.id": uid,
            "refit.step": step,
            "refit.phase": "cold" if step == 0 else "warm",
            "experiment": os.environ.get("MX_REFIT_EXPERIMENT", ""),
            "staging_mode": self.config.staging_mode,
        }

    def _clean(self, step: int) -> None:
        try:
            carrier = {}
            if self._cycle is not None:
                self._cycle.inject(carrier)
            with telemetry.extracted(carrier), telemetry.span("mx.refit.broadcast_cleanup", self._attributes(step)):
                super()._clean(step)
        except BaseException as error:
            if self._cycle is not None:
                self._cycle.finish(error)
            raise
        else:
            if self._cycle is not None:
                self._cycle.finish()
        finally:
            self._cycle = None

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
            self._set_cycle(step)
            with telemetry.extracted(self._carrier), telemetry.refit_attributes(self._attributes(step)):
                if self._trainer is None:
                    with telemetry.span("mx.refit.trainer_initialize", self._attributes(step)):
                        self._initialize(model)
                assert self._trainer is not None
                self._publish_and_wait(step, step_dir)
        except BaseException as error:
            if self._cycle is not None:
                self._cycle.finish(error)
            raise

    def _publish_and_wait(self, step: int, step_dir: Path) -> None:
        assert self._trainer is not None
        offered = [None, {}]
        if self.world.is_master:
            assert self._control is not None and self._mesh_id is not None
            version = self._control.create_weight_version(
                model_name=self.model_name,
                idempotency_key=uuid.uuid4().hex,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                trainer_mesh_id=self._mesh_id,
                version_number=step,
            )
            offered[0] = version.version_id
            telemetry.inject(offered[1])
            with telemetry.span("mx.refit.version_marker_publish", self._attributes(step, version.version_id)):
                marker = step_dir / VERSION_MARKER
                pending = marker.with_suffix(".pending")
                pending.write_text(json.dumps({"version_uid": version.version_id, "trace_context": offered[1]}))
                pending.replace(marker)
        broadcast_start = time.time_ns()
        dist.broadcast_object_list(offered, src=0)
        broadcast_end = time.time_ns()
        version = WeightVersionRef(offered[0])
        with (
            telemetry.extracted(offered[1]),
            telemetry.refit_attributes(self._attributes(step, version.version_id)),
            telemetry.span("mx.refit", self._attributes(step, version.version_id), start_time=broadcast_start),
        ):
            attributes = self._attributes(step, version.version_id)
            telemetry.completed_span("mx.refit.version_broadcast", broadcast_start, broadcast_end, attributes)
            if self._cycle is not None:
                self._cycle.set_attributes(attributes)
            with telemetry.span("mx.refit.publish", attributes):
                self._trainer.publish_version(version=version)

            # Installation acknowledgment is a PrimeRL outcome, not MX retirement.
            installed = step_dir / INSTALLED_MARKER
            deadline = time.monotonic() + self.timeout
            with telemetry.span(
                "mx.refit.wait_installed",
                {
                    **attributes,
                    "wait.marker": str(installed),
                    "wait.poll_interval_s": 0.1,
                },
            ):
                while not installed.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Inference did not install version {version.version_id} within {self.timeout}s"
                        )
                    time.sleep(0.1)
                if installed.read_text() != version.version_id:
                    raise RuntimeError("Inference acknowledged a different weight version")
            with telemetry.span("mx.refit.release", attributes):
                self._trainer.release_version(version=version)
            with telemetry.span("mx.refit.trainer_barrier", attributes):
                dist.barrier()


class ModelExpressWeightReceiver(WeightReceiver):
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
        receive_start = time.time_ns()
        self._ack(step)
        acknowledged = time.time_ns()
        marker = self.step_dir(step) / VERSION_MARKER
        await asyncio.wait_for(wait_for_path(marker, interval=0.01), timeout=self.config.timeout)
        marker_visible = time.time_ns()
        offered = json.loads(marker.read_text())
        uid = offered["version_uid"]
        carrier = offered["trace_context"]
        attributes = {
            "role": "orchestrator",
            "rank": 0,
            "step": step,
            "version_uid": uid,
            "refit.id": uid,
            "refit.step": step,
            "refit.phase": "cold" if step == 0 else "warm",
            "experiment": os.environ.get("MX_REFIT_EXPERIMENT", ""),
            "staging_mode": self.config.staging_mode,
        }
        with (
            telemetry.extracted(carrier),
            telemetry.refit_attributes(attributes),
            telemetry.span("mx.refit", attributes, start_time=receive_start),
        ):
            telemetry.completed_span("mx.refit.receiver_ack", receive_start, acknowledged, attributes)
            telemetry.completed_span(
                "mx.refit.wait_version_marker",
                acknowledged,
                marker_visible,
                {
                    **attributes,
                    "wait.marker": str(marker),
                    "wait.poll_interval_s": 0.01,
                },
            )
            await self._receive_version(step, uid)

    async def _receive_version(self, step: int, uid: str) -> None:
        deadline = time.monotonic() + self.config.timeout
        with telemetry.span("mx.refit.wait_version_ready"):
            while True:
                version = await asyncio.to_thread(self._control.get_weight_version, uid)
                if version.state is WeightVersionState.READY:
                    break
                if version.state is WeightVersionState.RELEASING:
                    raise RuntimeError(f"Weight version {uid} was retired before installation")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Weight version {uid} was not ready within {self.config.timeout}s")
                await asyncio.sleep(0.1)
        with telemetry.span("mx.refit.inference_update"):
            await self.admin_plane.update_weights(None, transport="modelexpress", step=step, version_uid=uid)
        await asyncio.to_thread(self._control.delete_weight_version, uid)
        with telemetry.span("mx.refit.installed_marker_publish"):
            installed = self.step_dir(step) / INSTALLED_MARKER
            pending = installed.with_suffix(".pending")
            pending.write_text(uid)
            pending.replace(installed)
