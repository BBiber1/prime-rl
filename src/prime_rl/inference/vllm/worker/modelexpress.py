"""vLLM extension using the public ModelExpress generator client."""

import atexit
from typing import TYPE_CHECKING, cast

import torch
from modelexpress_rl import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    VllmGeneratorContext,
    WeightSource,
    WeightVersionRef,
)
from torch import nn

from prime_rl.configs.inference import ModelExpressWeightBroadcastConfig

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
    ):
        from prime_rl.trainer.models import get_custom_causal_lm_cls
        from prime_rl.trainer.models.conversion_ops import apply_prime_to_hf

        hf_config = self.model_runner.model_config.hf_config
        chain = get_custom_causal_lm_cls(hf_config).conversion_chain(hf_config)
        config = ModelExpressWeightBroadcastConfig.model_validate(
            self.vllm_config.additional_config["weight_broadcast"]
        )
        self._worker_id = f"{session_id}:{rank_offset + self.rank}"
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
                staging_buffer_bytes=config.staging_buffer_bytes,
                staging_buffers_count=config.staging_buffers_count,
            )
        )
        atexit.register(self._generator.close)
        return self._worker_id

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None, version_uid: str | None = None):
        if version_uid is None:
            raise ValueError("modelexpress requires version_uid")
        version = WeightVersionRef(version_uid)
        staged = self._generator.stage_weight(version=version)
        try:
            self._generator.apply_weight(staged)
        finally:
            staged.release()
        return {"worker_id": self._worker_id, "version_uid": version_uid}
