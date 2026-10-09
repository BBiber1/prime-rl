"""Keep the framework adapter on ModelExpress's public client surface."""

import ast
import asyncio
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def refit_exporter(monkeypatch):
    pytest.importorskip("opentelemetry.sdk")
    from modelexpress import telemetry
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource({"service.name": "prime-rl-trainer"}))
    provider.add_span_processor(telemetry.RefitSpanProcessor(SimpleSpanProcessor(exporter)))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://unused/v1/traces")
    monkeypatch.setattr(telemetry, "_configured_pid", os.getpid())
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("modelexpress-boundary-test"))
    monkeypatch.setattr(telemetry, "_tracer_provider", provider)
    yield exporter
    provider.shutdown()


def _extract_function(path, name, *, class_name=None, namespace=None):
    source = ast.parse(path.read_text())
    owner = source.body
    if class_name is not None:
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        owner = cls.body
    function = next(
        node for node in owner if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    function.decorator_list = []
    module = ast.parse("from __future__ import annotations")
    module.body.append(function)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


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
                assert module == "modelexpress_rl"
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
    worker._rank = 3
    worker._generator = SimpleNamespace(stage_weight=stage_weight, apply_weight=apply_weight)
    if fail_install:
        with pytest.raises(RuntimeError, match="installation failed"):
            worker.update_weights_from_modelexpress(version_uid="version-a")
    else:
        result = worker.update_weights_from_modelexpress(version_uid="version-a")
        assert result is None
    assert calls == ["stage", "apply", "release"]


@pytest.mark.parametrize("fail_install", [False, True])
def test_modelexpress_admin_forwards_trace_and_preserves_failure(fail_install, refit_exporter):
    import httpx
    from modelexpress import telemetry
    from modelexpress_rl import RefitTrace
    from opentelemetry import trace

    from prime_rl.inference.vllm.worker.modelexpress import ModelExpressWeightUpdateWorker

    events = []
    worker = ModelExpressWeightUpdateWorker()
    worker._rank = 3
    staged = SimpleNamespace(release=lambda: events.append("release"))

    def stage_weight(*, version):
        assert version.version_id == "version-a"
        events.append("stage")
        return staged

    def apply_weight(handle):
        assert handle is staged
        with telemetry.span("install"):
            events.append("apply")
            if fail_install:
                raise RuntimeError("installation failed")

    worker._generator = SimpleNamespace(stage_weight=stage_weight, apply_weight=apply_weight)

    async def collective_rpc(method, *, args):
        assert method == "update_weights_from_modelexpress"
        assert args[0] == "version-a"
        assert {"traceparent", "baggage"} <= args[1].keys()
        assert worker.update_weights_from_modelexpress(*args) is None

    endpoint = _extract_function(
        ROOT / "src/prime_rl/inference/vllm/server.py",
        "update_weights_from_modelexpress",
        namespace={"engine_client": lambda _: SimpleNamespace(collective_rpc=collective_rpc)},
    )

    async def post(path, *, json, headers, timeout):
        assert path == "/update_weights_from_modelexpress"
        assert json == {"version_uid": "version-a"}

        async def body():
            return json

        assert await endpoint(SimpleNamespace(headers=headers, json=body)) == {"status": "ok"}
        return SimpleNamespace(raise_for_status=lambda: None)

    async def pause(clients, *, step):
        assert step == 5
        events.append("pause")

    async def resume(clients):
        events.append("resume")

    update = _extract_function(
        ROOT / "src/prime_rl/orchestrator/clients.py",
        "update_modelexpress_weights",
        class_name="AdminPlane",
        namespace={
            "asyncio": asyncio,
            "httpx": httpx,
            "UPDATE_WEIGHTS_TIMEOUT_S": 720,
            "_pause_engines": pause,
            "_resume_engines": resume,
        },
    )

    async def run_update():
        root = RefitTrace.trainer(step=5, rank=0, trainers=1, generators=1, staging_mode="broadcast")
        root.set_version("version-a")
        orchestrator = RefitTrace.orchestrator(step=5, staging_mode="broadcast")
        orchestrator.bind(root.context()["root"], version_uid="version-a")
        admin = SimpleNamespace(clients=[SimpleNamespace(post=post)], _modelexpress_lock=asyncio.Lock())
        expected = pytest.raises(RuntimeError, match="installation failed") if fail_install else nullcontext()
        with expected, root, orchestrator, orchestrator.span("inference_update"):
            await update(admin, version_uid="version-a", trace=orchestrator, step=5)
        assert not trace.get_current_span().get_span_context().is_valid

    asyncio.run(run_update())
    assert events == ["pause", "stage", "apply", "release"] + ([] if fail_install else ["resume"])
    spans = {span.name: span for span in refit_exporter.get_finished_spans()}
    parents = {
        "mx.refit.orchestrator": "mx.refit.cycle",
        "mx.refit.inference_update": "mx.refit.orchestrator",
        "mx.refit.pause_engines": "mx.refit.inference_update",
        "mx.refit.update_weights_rpc": "mx.refit.inference_update",
        "mx.refit.generators": "mx.refit.cycle",
        "mx.refit.generator": "mx.refit.generators",
        "install": "mx.refit.generator",
    }
    if not fail_install:
        parents["mx.refit.resume_engines"] = "mx.refit.inference_update"
    for child_name, parent_name in parents.items():
        child, parent = spans[child_name], spans[parent_name]
        assert child.parent.span_id == parent.context.span_id
        assert child.context.trace_id == parent.context.trace_id
        assert parent.start_time <= child.start_time <= child.end_time <= parent.end_time
    assert ("mx.refit.resume_engines" in spans) is not fail_install
    assert spans["mx.refit.cycle"].resource.attributes["service.name"] == "root"
    for name in ("mx.refit.generator", "install"):
        assert spans[name].attributes["rank"] == 3
        assert spans[name].attributes["role"] == "generator"
        assert spans[name].attributes["step"] == 5
        assert spans[name].attributes["version_uid"] == "version-a"
    for name in ("mx.refit.orchestrator", "mx.refit.generators", "mx.refit.generator"):
        assert spans[name].attributes["status"] == ("failed" if fail_install else "complete")


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


@pytest.mark.parametrize("outcome", ["success", "timeout", "wrong_uid", "read_error"])
def test_modelexpress_installation_outcome_reaches_non_master_without_shared_storage(tmp_path, outcome):
    source = ast.parse((ROOT / "src/prime_rl/transports/weights/modelexpress.py").read_text())
    sender = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ModelExpressWeightSender"
    )
    wait_installed = next(
        node for node in sender.body if isinstance(node, ast.FunctionDef) and node.name == "_wait_installed"
    )
    # Execute the real acknowledgment block without importing the GPU runtime.
    module = ast.parse("def acknowledge(self, version, step_dir, trace): pass")
    module.body[0].body = wait_installed.body
    messages = []
    rank = 0

    def broadcast_object_list(values, src):
        assert src == 0
        assert len(values) == 1 and isinstance(values[0], bool)
        if rank == 0:
            messages.append(pickle.dumps(values))
        else:
            values[:] = pickle.loads(messages[0])

    namespace = {
        "time": time,
        "dist": SimpleNamespace(
            broadcast_object_list=broadcast_object_list,
        ),
        "INSTALLED_MARKER": ".installed",
    }
    exec(compile(ast.fix_missing_locations(module), "<installation acknowledgment>", "exec"), namespace)
    installed = tmp_path / ".installed"
    if outcome == "read_error":
        installed.mkdir()
    elif outcome != "timeout":
        installed.write_text("version-a" if outcome == "success" else "version-b")
    error_type = {"timeout": TimeoutError, "wrong_uid": RuntimeError, "read_error": IsADirectoryError}
    errors = []
    for rank in (0, 1):
        sender = SimpleNamespace(
            world=SimpleNamespace(is_master=rank == 0),
            timeout=0,
        )
        step_dir = tmp_path if rank == 0 else tmp_path / "non_master_local_fs"
        if outcome == "success":
            namespace["acknowledge"](
                sender, SimpleNamespace(version_id="version-a"), step_dir, SimpleNamespace(span=lambda _: nullcontext())
            )
        else:
            expected_error = error_type[outcome] if rank == 0 else RuntimeError
            with pytest.raises(expected_error) as exc:
                namespace["acknowledge"](
                    sender,
                    SimpleNamespace(version_id="version-a"),
                    step_dir,
                    SimpleNamespace(span=lambda _: nullcontext()),
                )
            errors.append(str(exc.value))
    assert len(messages) == 1
    if errors:
        assert errors[1] == "Inference weight installation failed on trainer rank zero"
