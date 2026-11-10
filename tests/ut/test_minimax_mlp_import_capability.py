# SPDX-License-Identifier: Apache-2.0
"""Run the whole upstream model's MLP across the NVIDIA import boundary."""

import ast
import importlib.util
import sys
from types import SimpleNamespace as NS

import pytest


def load_model(modules, monkeypatch, source, ppu):
    """Configure only external imports; execute the complete real model module."""
    path = source / "vllm/models/minimax_m3/nvidia/model.py"

    def unavailable(self, *args, **kwargs):
        raise AssertionError("Unconfigured external dependency was invoked")

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.module.startswith(
            ("vllm", "transformers")
        ):
            attrs = {
                alias.name: type(alias.name, (), {"__init__": unavailable})
                for alias in node.names
            }
            modules(node.module, **attrs)
    modules("vllm", _custom_ops=modules("vllm._custom_ops"))
    modules("vllm.envs")

    class Module:
        def __init__(self):
            self.initialized = True

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    tensor = type("Tensor", (), {})
    nn = modules("torch.nn", Module=Module, Parameter=tensor, ModuleList=list)
    modules(
        "torch",
        nn=nn,
        Tensor=tensor,
        dtype=type("dtype", (), {}),
        device=type("device", (), {}),
    )
    modules("vllm.logger", init_logger=lambda name: NS())
    modules(
        "vllm.multimodal",
        MULTIMODAL_REGISTRY=NS(
            register_processor=lambda processor, *, info, dummy_inputs: lambda cls: cls
        ),
    )
    modules(
        "vllm.model_executor.models.utils",
        WeightsMapper=lambda *, orig_to_new_prefix, orig_to_new_substr: NS(),
    )
    modules(
        "vllm.compilation.breakable_cudagraph",
        eager_break_during_capture=lambda fn: fn,
    )
    modules(
        "vllm.platforms", current_platform=NS(is_ppu=lambda: ppu, is_cuda=lambda: True)
    )
    name = "vllm.models.minimax_m3.nvidia.model"
    modules(name)
    spec = importlib.util.spec_from_file_location(name, path)
    provider = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, provider)
    monkeypatch.setattr(sys.modules[name.rpartition(".")[0]], "model", provider)
    spec.loader.exec_module(provider)
    return provider


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
@pytest.mark.parametrize("reduce_results", [False, True])
def test_release_mlp_keeps_constructor_and_forward_contract(
    modules, monkeypatch, upstream_source_root, ppu, reduce_results
):
    provider = load_model(modules, monkeypatch, upstream_source_root, ppu)
    calls = []
    quant = object()
    gate_up, activated, output = object(), object(), object()

    class Linear:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs

        def __call__(self, x):
            calls.append(("linear-forward", self, x))
            return (gate_up if isinstance(self.args[1], list) else output), None

    class Activation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, x):
            assert x is gate_up
            calls.append(("act", x))
            return activated

    provider.MergedColumnParallelLinear = provider.RowParallelLinear = Linear
    provider.SiluAndMulWithClamp = Activation
    config = NS(
        hidden_size=8,
        hidden_act="swigluoai",
        swiglu_limit=7,
        swiglu_alpha=1.702,
        swiglu_beta=1,
    )
    mlp = provider.MiniMaxM3MLP(config, 16, quant, reduce_results, "layers.0.mlp")
    assert mlp.initialized
    assert mlp.gate_up_proj.args == (8, [16, 16])
    assert mlp.gate_up_proj.kwargs == dict(
        bias=False, quant_config=quant, prefix="layers.0.mlp.gate_up_proj"
    )
    assert mlp.down_proj.args == (16, 8)
    assert mlp.down_proj.kwargs == dict(
        bias=False,
        quant_config=quant,
        reduce_results=reduce_results,
        prefix="layers.0.mlp.down_proj",
    )
    assert mlp.act_fn.kwargs == dict(swiglu_limit=7, alpha=1.702, beta=1)
    x = NS(shape=(64, 8))
    assert mlp(x) is output
    assert calls == [
        ("linear-forward", mlp.gate_up_proj, x),
        ("act", gate_up),
        ("linear-forward", mlp.down_proj, activated),
    ]


@pytest.mark.upstream_source
@pytest.mark.parametrize("ppu", [False, True])
def test_release_mlp_retains_activation_validation(
    modules, monkeypatch, upstream_source_root, ppu
):
    provider = load_model(modules, monkeypatch, upstream_source_root, ppu)
    provider.MergedColumnParallelLinear = provider.RowParallelLinear = lambda *a, **k: (
        object()
    )
    with pytest.raises(ValueError, match="Only swigluoai is supported"):
        provider.MiniMaxM3MLP(NS(hidden_size=8, hidden_act="unsupported"), 16)
