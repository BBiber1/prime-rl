"""vLLM extension using the public ModelExpress generator client."""

import atexit
from typing import TYPE_CHECKING, cast

import torch
from modelexpress import telemetry
from modelexpress_rl import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    VllmGeneratorContext,
    WeightSource,
    WeightVersionRef,
)
from torch import nn

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object


class ModelExpressWeightUpdateWorker(Worker):
    def liveness_probe(self) -> None:
        return None

    def init_broadcaster(
        self,
        host,
        port,
        rank_offset,
        inference_world_size,
        timeout,
        session_id="default",
        staging_buffer_bytes=None,
        staging_buffers_count=1,
    ):
        from prime_rl.trainer.models import get_custom_causal_lm_cls
        from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

        hf_config = self.model_runner.model_config.hf_config
        chain = get_custom_causal_lm_cls(hf_config).conversion_chain(hf_config)
        self._rank = rank_offset + self.rank
        self._worker_id = f"{session_id}:{self._rank}"
        self._generator = ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=VllmGeneratorContext(
                    model=cast(nn.Module, self.model_runner.get_model()),
                    vllm_config=self.vllm_config,
                    convert_native_to_hf=(lambda tensors: apply_prime_to_hf(tensors, chain)) if chain else None,
                ),
                model_name=self.model_runner.model_config.model,
                server_url=f"{host}:{port}",
                source_order=(WeightSource.TRAINER,),
                worker_id=self._worker_id,
                staging_buffer_bytes=staging_buffer_bytes,
                staging_buffers_count=staging_buffers_count,
            )
        )
        atexit.register(self._generator.close)
        return self._worker_id

    @torch.no_grad()
    def update_weights_from_path(
        self,
        weight_dir: str | None = None,
        version_uid: str | None = None,
        trace_context: dict | None = None,
        step: int = 0,
    ):
        if version_uid is None:
            raise ValueError("modelexpress requires version_uid")
        telemetry.configure("prime-rl-inference")
        with (
            telemetry.extracted(trace_context or {}),
            telemetry.refit_attributes(
                {"step": step, "refit.step": step, "version_uid": version_uid, "refit.id": version_uid},
                role="generator",
                rank=getattr(self, "_rank", 0),
            ),
            telemetry.refit_span("mx.refit.generator") as generator,
        ):
            version = WeightVersionRef(version_uid)
            staged = self._generator.stage_weight(version=version)
            try:
                self._generator.apply_weight(staged)
            finally:
                staged.release()
        result = {"worker_id": self._worker_id, "version_uid": version_uid}
        if telemetry.enabled():
            result["refit_timing"] = generator.interval
        return result
