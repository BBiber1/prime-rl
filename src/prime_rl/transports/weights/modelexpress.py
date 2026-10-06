"""FSDP weight updates through ModelExpress's public clients."""

import asyncio
import atexit
import os
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import zmq
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


class ModelExpressWeightSender(WeightSender):
    filesystem_handshake = False

    def __init__(self, output_dir: Path, config: ModelExpressWeightBroadcastConfig, model_name: str):
        super().__init__(output_dir, config.timeout)
        self.config = config
        self.model_name = model_name
        self._trainer: ModelExpressTrainerClient | None = None
        self._control: ModelExpressControlClient | None = None
        self._mesh_id: str | None = None
        self._coordination = None

    def _attributes(self, step: int) -> dict:
        return {
            "step": step,
            "refit.step": step,
            "refit.phase": "cold" if step == 0 else "warm",
            "experiment": os.environ.get("MX_REFIT_EXPERIMENT", ""),
            "staging_mode": self.config.staging_mode,
        }

    def _reply(self) -> dict:
        if not self._coordination.poll(int(self.timeout * 1000)):
            raise TimeoutError(f"No ModelExpress receiver acknowledgment within {self.timeout}s")
        return self._coordination.recv_json()

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
        telemetry.configure("prime-rl-trainer")
        attributes = self._attributes(step)
        cycle = trainers = trainer = None
        root_carrier, trainer_carrier = {}, {}
        started = time.time_ns()
        error = None
        try:
            if self.world.is_master:
                cycle = telemetry.RefitCycle({**attributes, "refit.root": True})
                cycle.inject(root_carrier)
                trainers = telemetry.RefitCycle(
                    {**attributes, "role": "trainer"}, name="mx.refit.trainers", parent=root_carrier
                )
                trainers.inject(trainer_carrier)
                trainer = telemetry.RefitCycle(
                    {**attributes, "role": "trainer", "rank": self.world.rank},
                    name="mx.refit.trainer",
                    parent=trainer_carrier,
                )
                if self._coordination is None:
                    self._coordination = zmq.Context.instance().socket(zmq.REQ)
                    self._coordination.setsockopt(zmq.LINGER, 0)
                    self._coordination.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
                    self._coordination.connect(f"tcp://{self.config.host}:{self.config.coordination_port}")
                    atexit.register(self._coordination.close)
                with trainer.active(), telemetry.refit_attributes(attributes, role="trainer", rank=self.world.rank):
                    with telemetry.span("mx.refit.wait_receiver_ready"):
                        self._coordination.send_json({"step": step, "trace_context": root_carrier})
                        if self._reply() != {"ready": step}:
                            raise RuntimeError("Receiver acknowledged a different weight offer")
            carriers = [root_carrier, trainer_carrier]
            dist.broadcast_object_list(carriers, src=0)
            broadcast_end = time.time_ns()
            root_carrier, trainer_carrier = carriers
            if trainer is None:
                trainer = telemetry.RefitCycle(
                    {**attributes, "role": "trainer", "rank": self.world.rank},
                    name="mx.refit.trainer",
                    parent=trainer_carrier,
                )
            with trainer.active(), telemetry.refit_attributes(attributes, role="trainer", rank=self.world.rank):
                telemetry.completed_span("mx.refit.trace_context_broadcast", started, broadcast_end)
                if self._trainer is None:
                    with telemetry.span("mx.refit.trainer_initialize"):
                        self._initialize(model)
                assert self._trainer is not None
                offered = [None]
                broadcast_start = time.time_ns()
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
                    uid_attributes = {"version_uid": offered[0], "refit.id": offered[0]}
                    cycle.set_attributes(uid_attributes)
                    trainers.set_attributes(uid_attributes)
                    cycle.inject(root_carrier)
                    with telemetry.span("mx.refit.version_offer"):
                        self._coordination.send_json({"version_uid": offered[0], "trace_context": root_carrier})
                dist.broadcast_object_list(offered, src=0)
                broadcast_end = time.time_ns()
                uid_attributes = {"version_uid": offered[0], "refit.id": offered[0]}
                trainer.set_attributes(uid_attributes)
                with telemetry.refit_attributes(uid_attributes):
                    telemetry.completed_span("mx.refit.version_broadcast", broadcast_start, broadcast_end)
                    version = WeightVersionRef(offered[0])
                    with telemetry.span("mx.refit.publish"):
                        self._trainer.publish_version(version=version)
                    installed = [None]
                    with telemetry.span("mx.refit.wait_installed"):
                        if self.world.is_master:
                            installed[0] = self._reply()
                        dist.broadcast_object_list(installed, src=0)
                        if installed[0]["version_uid"] != version.version_id:
                            raise RuntimeError("Inference acknowledged a different weight version")
                    with telemetry.span("mx.refit.release"):
                        self._trainer.release_version(version=version)
                    with telemetry.span("mx.refit.trainer_barrier"):
                        dist.barrier()
            trainer.finish()
            if telemetry.enabled():
                intervals = [None] * self.world.world_size
                dist.all_gather_object(intervals, trainer.interval)
                if self.world.is_master:
                    for interval in intervals:
                        trainers.include(interval)
                    for interval in installed[0]["intervals"]:
                        cycle.include(interval)
        except BaseException as failure:
            error = failure
            raise
        finally:
            for envelope in (trainer, trainers, cycle):
                if envelope is not None:
                    envelope.finish(error)


class ModelExpressWeightReceiver(WeightReceiver):
    async def initialize(self) -> None:
        self._offer = None
        self._coordination = zmq.Context.instance().socket(zmq.REP)
        self._coordination.setsockopt(zmq.LINGER, 0)
        self._coordination.setsockopt(zmq.SNDTIMEO, int(self.config.timeout * 1000))
        self._coordination.bind(f"tcp://*:{self.config.coordination_port}")
        atexit.register(self._coordination.close)
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

    def _poll_offer(self) -> None:
        if self._offer is None and self._coordination.poll(0):
            self._offer = self._coordination.recv_json()

    def available_versions(self) -> list[int]:
        self._poll_offer()
        return [self._offer["step"]] if self._offer is not None else []

    def is_published(self, step: int) -> bool:
        self._poll_offer()
        return self._offer is not None and self._offer["step"] == step

    def next_version(self, current: int) -> int:
        self._poll_offer()
        return max(current, self._offer["step"]) if self._offer is not None else current

    async def wait_published(self, step: int, cancelled=None) -> None:
        deadline = time.monotonic() + self.config.timeout
        while not self.is_published(step):
            if cancelled is not None and cancelled():
                raise asyncio.CancelledError
            if time.monotonic() >= deadline:
                raise TimeoutError(f"No ModelExpress weight offer within {self.config.timeout}s")
            await asyncio.sleep(0.1)

    async def receive(self, step: int) -> None:
        telemetry.configure("prime-rl-orchestrator")
        if not self.is_published(step):
            raise RuntimeError(f"No ModelExpress weight offer for step {step}")
        carrier = self._offer["trace_context"]
        with (
            telemetry.extracted(carrier),
            telemetry.refit_attributes(role="orchestrator", rank=0),
            telemetry.refit_span("mx.refit.orchestrator") as orchestrator,
        ):
            with telemetry.span("mx.refit.receiver_ack"):
                self._coordination.send_json({"ready": step})
                self._offer = None
            deadline = time.monotonic() + self.config.timeout
            with telemetry.span("mx.refit.wait_version_offer"):
                while not self._coordination.poll(0):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"No ModelExpress version within {self.config.timeout}s")
                    await asyncio.sleep(0.01)
                offered = self._coordination.recv_json()
            uid = offered["version_uid"]
            uid_attributes = {"version_uid": uid, "refit.id": uid}
            orchestrator.set_attributes(uid_attributes)
            with telemetry.refit_attributes(uid_attributes):
                generators_interval = await self._receive_version(step, uid, offered["trace_context"])
        self._coordination.send_json({"version_uid": uid, "intervals": [orchestrator.interval, generators_interval]})

    async def _receive_version(self, step: int, uid: str, carrier: dict) -> list[int] | None:
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
            interval = await self.admin_plane.update_weights(
                None, transport="modelexpress", step=step, version_uid=uid, trace_context=carrier
            )
        with telemetry.span("mx.refit.retire"):
            await asyncio.to_thread(self._control.delete_weight_version, uid)
        return interval
