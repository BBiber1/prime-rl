"""Keep the framework adapter on ModelExpress's public client surface."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/prime_rl/transports/weights/modelexpress.py",
        "src/prime_rl/inference/vllm/worker/modelexpress.py",
    ],
)
def test_modelexpress_adapter_uses_only_public_clients(relative_path):
    source = ast.parse((ROOT / relative_path).read_text())
    client_methods = {
        "_trainer": {"bind_tensors", "publish_version", "release_version", "close"},
        "_control": {
            "create_trainer_mesh",
            "create_weight_version",
            "get_weight_version",
            "delete_weight_version",
            "delete_trainer_mesh",
            "close",
        },
        "_generator": {"stage_weight", "apply_weight", "close"},
    }
    for node in ast.walk(source):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("modelexpress"):
                assert module == "modelexpress_rl" or (
                    module == "modelexpress" and [name.name for name in node.names] == ["telemetry"]
                )
                assert all(not name.name.startswith("_") for name in node.names)
            assert not module.startswith(("grpc", "nixl"))
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith(("modelexpress", "grpc", "nixl")) for name in node.names)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            owner = node.value
            if isinstance(owner.value, ast.Name) and owner.value.id == "self" and owner.attr in client_methods:
                assert not node.attr.startswith("_")
                if isinstance(node.ctx, ast.Load) and node.attr in client_methods[owner.attr]:
                    continue
                assert (owner.attr, node.attr) == ("_trainer", "worker_id")


@pytest.mark.parametrize("fail_install", [False, True])
def test_modelexpress_worker_uses_stage_apply_release(fail_install):
    from prime_rl.inference.vllm.worker.modelexpress import ModelExpressWeightUpdateWorker

    calls = []
    staged = SimpleNamespace(release=lambda: calls.append("release"))

    def stage_weight(*, version):
        assert version.version_id == "version-a"
        calls.append("stage")
        return staged

    def apply_weight(handle):
        assert handle is staged
        calls.append("apply")
        if fail_install:
            raise RuntimeError("installation failed")

    worker = ModelExpressWeightUpdateWorker()
    worker._worker_id = "worker-0"
    worker._generator = SimpleNamespace(stage_weight=stage_weight, apply_weight=apply_weight)
    if fail_install:
        with pytest.raises(RuntimeError, match="installation failed"):
            worker.update_weights_from_path(version_uid="version-a")
    else:
        result = worker.update_weights_from_path(version_uid="version-a")
        assert result == {"worker_id": "worker-0", "version_uid": "version-a"}
    assert calls == ["stage", "apply", "release"]


def test_modelexpress_worker_forwards_generator_buffer_config():
    source = ast.parse((ROOT / "src/prime_rl/inference/vllm/worker/modelexpress.py").read_text())
    config = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelExpressGeneratorConfig"
    )
    keywords = {item.arg: item.value for item in config.keywords}
    for name in ("staging_buffer_bytes", "staging_buffers_count"):
        assert isinstance(keywords[name], ast.Name)
        assert keywords[name].id == name


def test_modelexpress_worker_preserves_otlp_parent(monkeypatch):
    import os

    from modelexpress import telemetry
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from prime_rl.inference.vllm.worker.modelexpress import ModelExpressWeightUpdateWorker

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(telemetry.RefitSpanProcessor(SimpleSpanProcessor(exporter)))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused/v1/traces")
    monkeypatch.setattr(telemetry, "_configured_pid", os.getpid())
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("worker-test"))
    cycle = telemetry.RefitCycle({"refit.root": True, "step": 4})
    carrier = {}
    cycle.inject(carrier)
    worker = ModelExpressWeightUpdateWorker()
    worker._worker_id = "worker-3"
    worker._rank = 3
    staged = SimpleNamespace(release=lambda: None)
    worker._generator = SimpleNamespace(stage_weight=lambda **_kwargs: staged, apply_weight=lambda _handle: None)
    try:
        assert worker.update_weights_from_path(version_uid="version-a", trace_context=carrier, step=4) == {
            "worker_id": "worker-3",
            "version_uid": "version-a",
        }
        cycle.finish()
        spans = exporter.get_finished_spans()
        root = next(span for span in spans if span.name == "mx.refit.cycle")
        refit = next(span for span in spans if span.name == "mx.refit.generator")
        assert refit.context.trace_id == root.context.trace_id
        assert refit.parent.span_id == root.context.span_id
        assert refit.attributes["role"] == "generator" and refit.attributes["rank"] == 3
        assert refit.attributes["step"] == 4 and refit.attributes["version_uid"] == "version-a"
    finally:
        provider.shutdown()
