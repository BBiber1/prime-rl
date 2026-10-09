"""FSDP weight updates through ModelExpress's public clients."""

import asyncio
import atexit
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from modelexpress_rl import (
    FSDPTrainerContext,
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    RefitTrace,
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
    def __init__(self, output_dir: Path, config: ModelExpressWeightBroadcastConfig, *, model_name: str):
        super().__init__(output_dir, config.timeout)
        self.config = config
        self.model_name = model_name
        self._trainer: ModelExpressTrainerClient | None = None
        self._control: ModelExpressControlClient | None = None
        self._mesh_id: str | None = None
        self._trace: RefitTrace | None = None

    def _start_trace(self, step: int) -> RefitTrace:
        return RefitTrace.trainer(
            step=step,
            rank=self.world.rank,
            trainers=self.world.world_size,
            generators=self.config.inference_world_size,
            staging_mode=self.config.staging_mode,
        )

    def _wait_for_receiver_ready(self, step_dir: Path) -> None:
        self._trace = self._start_trace(int(step_dir.name.removeprefix("step_")))
        with self._trace.span("wait_receiver_ready"):
            super()._wait_for_receiver_ready(step_dir)

    def _clean(self, step: int) -> None:
        with self._trace, self._trace.span("broadcast_cleanup"):
            super()._clean(step)

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
        trace = self._trace if self.world.is_master else self._start_trace(step)
        if self._trainer is None:
            with trace.span("trainer_initialize"):
                self._initialize(model)
        assert self._trainer is not None
        offered = [None, None]
        with trace.active():
            if self.world.is_master:
                assert self._control is not None and self._mesh_id is not None
                version = self._control.create_weight_version(
                    model_name=self.model_name,
                    idempotency_key=uuid.uuid4().hex,
                    payload_format=WeightPayloadFormat.FULL_TENSOR,
                    trainer_mesh_id=self._mesh_id,
                    version_number=step,
                    trace_context=trace.context()["root"],
                )
                trace.set_version(version.version_id)
                offered = [version.version_id, trace.context()]
                with trace.span("version_marker_publish"):
                    marker = step_dir / VERSION_MARKER
                    pending = marker.with_suffix(".pending")
                    pending.write_text(version.version_id)
                    pending.replace(marker)
        with trace.span("version_broadcast"):
            dist.broadcast_object_list(offered, src=0)
        version = WeightVersionRef(offered[0])
        trace.bind(offered[1]["trainers"], version_uid=version.version_id)
        with trace.active():
            self._trainer.publish_version(version=version)
            self._wait_installed(version, step_dir, trace)
            self._trainer.release_version(version=version)
            with trace.span("trainer_barrier"):
                dist.barrier()
        if not self.world.is_master:
            trace.finish()

    def _wait_installed(self, version: WeightVersionRef, step_dir: Path, trace: RefitTrace) -> None:
        # Installation acknowledgment is a PrimeRL outcome, not MX retirement.
        installation_failed = [False]
        installation_error = None
        if self.world.is_master:
            try:
                with trace.span("wait_installed"):
                    installed = step_dir / INSTALLED_MARKER
                    deadline = time.monotonic() + self.timeout
                    while not installed.exists():
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Inference did not install version {version.version_id} within {self.timeout}s"
                            )
                        time.sleep(0.1)
                    if installed.read_text() != version.version_id:
                        raise RuntimeError("Inference acknowledged a different weight version")
            except Exception as exc:
                installation_failed[0] = True
                installation_error = exc
        dist.broadcast_object_list(installation_failed, src=0)
        if installation_failed[0]:
            if installation_error is not None:
                raise installation_error
            raise RuntimeError("Inference weight installation failed on trainer rank zero")


class ModelExpressWeightReceiver(WeightReceiver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace: RefitTrace | None = None
        self._trace_step: int | None = None

    def _start_trace(self, step: int) -> RefitTrace:
        if self._trace is not None:
            self._trace.finish()
        self._trace_step = step
        self._trace = RefitTrace.orchestrator(step=step, staging_mode=self.config.staging_mode)
        return self._trace

    async def wait_published(self, step: int, cancelled=None) -> None:
        trace = self._start_trace(step)
        try:
            with trace.span("wait_published"):
                await super().wait_published(step, cancelled=cancelled)
        except BaseException:
            self._trace = None
            raise

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
        trace = self._trace if self._trace is not None and self._trace_step == step else self._start_trace(step)
        self._trace = None
        with trace:
            with trace.span("receiver_ack"):
                self._ack(step)
            marker = self.step_dir(step) / VERSION_MARKER
            with trace.span("wait_version_marker"):
                await asyncio.wait_for(wait_for_path(marker, interval=0.01), timeout=self.config.timeout)
                uid = marker.read_text()
            deadline = time.monotonic() + self.config.timeout
            with trace.span("version_context_lookup"):
                version = await asyncio.to_thread(self._control.get_weight_version, uid)
            trace.bind(version.trace_context, version_uid=uid)
            with trace.span("wait_version_ready"):
                while True:
                    if version.state is WeightVersionState.READY:
                        break
                    if version.state is WeightVersionState.RELEASING:
                        raise RuntimeError(f"Weight version {uid} was retired before installation")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Weight version {uid} was not ready within {self.config.timeout}s")
                    await asyncio.sleep(0.1)
                    version = await asyncio.to_thread(self._control.get_weight_version, uid)
            with trace.span("inference_update"):
                await self.admin_plane.update_modelexpress_weights(version_uid=uid, step=step, trace=trace)
            with trace.active():
                await asyncio.to_thread(self._control.delete_weight_version, uid)
            with trace.span("installed_marker_publish"):
                installed = self.step_dir(step) / INSTALLED_MARKER
                pending = installed.with_suffix(".pending")
                pending.write_text(uid)
                pending.replace(installed)
