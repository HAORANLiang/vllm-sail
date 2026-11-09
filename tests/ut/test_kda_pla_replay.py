# SPDX-License-Identifier: Apache-2.0
"""PLA replay serves Kimi KDA RecoverSSM verify and commit on PPU."""

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tests.support.source import ROOT, definition, function
from tests.ut.test_kda_workspace_reservation import configs as constructor_configs
from tests.ut.test_kda_workspace_reservation import kda_constructor as _kda_constructor
from tests.ut.test_ppu_kernel_capabilities import load_patch
from tests.ut.test_ppu_kernel_capabilities import modules as _modules

kda_constructor = _kda_constructor
modules = _modules

KDA = "vllm_sail/patch/enhancement/attention/kda.py"
REPLAY = "pla.decode.kda_replayssm_vllm"
SDK = "vllm_sail.attention.pla_kda"
LOGGER = NS(info_once=lambda *args: None)


class HostTensor:
    """Records the view chain the wrapper builds on a host tensor double."""

    def __init__(self, name, ops=(), shape=(5,)):
        self.name, self.ops, self.shape = name, ops, shape

    def __getitem__(self, index):
        return HostTensor(self.name, (*self.ops, ("index", index)))

    def flatten(self, dim):
        return HostTensor(self.name, (*self.ops, ("flatten", dim)))

    def new_empty(self, shape):
        return HostTensor("allocated", (("new_empty", shape),))


def replay_verify(monkeypatch, calls, upstream_calls=None):
    pla = ModuleType(REPLAY)
    pla.kda_replay_verify = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, REPLAY, pla)
    return function(
        KDA,
        "_recoverssm_verify",
        {
            "_upstream_recoverssm_verify": lambda *args: upstream_calls.append(args),
            "logger": LOGGER,
            "_replay_conv_state_layout": lambda: "DS",
            "_kda_metadata": NS(NULL_BLOCK_ID=0),
        },
    )


def replay_layer(records="records"):
    return NS(
        _pla_replay_records=records,
        kv_cache=("conv_state", "checkpoint_state", "workspace"),
        A_log="A_log",
        dt_bias="dt_bias",
        gate_lower_bound=-5.0,
        local_num_heads=12,
        head_dim=128,
    )


def spec_metadata():
    return NS(spec_state_indices_tensor=HostTensor("state_indices"), num_spec_decodes=2)


def test_verify_maps_upstream_arguments_onto_pla_replay(monkeypatch):
    calls = []
    fn = replay_verify(monkeypatch, calls)
    layer = replay_layer()
    mixed, out = HostTensor("mixed_qkv"), HostTensor("out")
    result = fn(
        layer,
        spec_metadata(),
        mixed,
        HostTensor("raw_g"),
        HostTensor("raw_beta"),
        "conv_weight",
        "query_start_loc",
        out,
    )

    assert result is out
    assert len(calls) == 1
    call = calls[0]
    assert call["mixed_qkv"] is mixed
    assert (call["raw_g"].name, call["raw_g"].ops) == (
        "raw_g",
        (("index", 0), ("flatten", -2)),
    )
    assert (call["raw_beta"].name, call["raw_beta"].ops) == (
        "raw_beta",
        (("index", 0),),
    )
    assert (call["out"].name, call["out"].ops) == (
        "out",
        (("index", 0), ("flatten", -2)),
    )
    assert (call["state_indices"].name, call["state_indices"].ops) == (
        "state_indices",
        (("index", (slice(None, 2), 0)),),
    )
    assert {
        key: call[key]
        for key in (
            "conv_weight",
            "conv_state",
            "checkpoint_state",
            "A_log",
            "dt_bias",
            "lower_bound",
            "records",
            "query_start_loc",
            "conv_state_layout",
            "null_block_id",
        )
    } == {
        "conv_weight": "conv_weight",
        "conv_state": "conv_state",
        "checkpoint_state": "checkpoint_state",
        "A_log": "A_log",
        "dt_bias": "dt_bias",
        "lower_bound": -5.0,
        "records": "records",
        "query_start_loc": "query_start_loc",
        "conv_state_layout": "DS",
        "null_block_id": 0,
    }
    assert "scale" not in call


def test_verify_allocates_the_output_when_the_caller_passes_none(monkeypatch):
    calls = []
    fn = replay_verify(monkeypatch, calls)
    args = (HostTensor("raw_g"), HostTensor("raw_beta"), "conv_weight", "qsl")
    result = fn(replay_layer(), spec_metadata(), HostTensor("mixed"), *args, None)

    assert (result.name, result.ops) == ("allocated", (("new_empty", (1, 5, 12, 128)),))
    assert calls[0]["out"].ops == (
        ("new_empty", (1, 5, 12, 128)),
        ("index", 0),
        ("flatten", -2),
    )


def test_verify_without_pla_records_delegates_to_upstream(monkeypatch):
    calls, upstream_calls = [], []
    fn = replay_verify(monkeypatch, calls, upstream_calls)
    layer = replay_layer(records=None)
    args = tuple(object() for _ in range(7))
    fn(layer, *args)

    assert calls == []
    assert upstream_calls == [(layer, *args)]


@pytest.mark.parametrize("dim_first,layout", [(True, "DS"), (False, "SD")])
def test_replay_names_the_raw_conv_state_layout(dim_first, layout):
    fn = function(
        KDA,
        "_replay_conv_state_layout",
        {"is_conv_state_dim_first": lambda: dim_first},
    )
    assert fn() == layout


def commit_builder(layers, context=None):
    return NS(
        recoverssm_context=context,
        layer_names=list(layers),
        vllm_config=NS(
            compilation_config=NS(static_forward_context=dict(layers)),
            num_speculative_tokens=3,
            scheduler_config=NS(max_num_seqs=8),
        ),
    )


def commit_context(monkeypatch, created, upstream_calls):
    class Context:
        @classmethod
        def create(cls, layers, **kwargs):
            created.append((layers, kwargs))
            return "pla-context"

    sdk = ModuleType(SDK)
    sdk.PLAKDAReplayCommitContext = Context
    monkeypatch.setitem(sys.modules, SDK, sdk)
    return function(
        KDA,
        "_get_recoverssm_context",
        {
            "_upstream_get_recoverssm_context": lambda self: (
                upstream_calls.append(self) or "upstream-context"
            ),
            "logger": LOGGER,
            "_replay_conv_state_layout": lambda: "SD",
            "_kda_metadata": NS(NULL_BLOCK_ID=0),
        },
    )


def test_commit_context_runs_pla_replay_for_pla_layers(monkeypatch):
    created, upstream_calls = [], []
    fn = commit_context(monkeypatch, created, upstream_calls)
    layers = {"a": NS(_pla_replay_records="ra"), "b": NS(_pla_replay_records="rb")}
    builder = commit_builder(layers)

    assert fn(builder) == "pla-context"
    assert fn(builder) == "pla-context"
    assert builder.recoverssm_context == "pla-context"
    assert upstream_calls == []
    assert created == [
        (
            [layers["a"], layers["b"]],
            dict(
                spec_query_len=4,
                max_num_reqs=8,
                conv_state_layout="SD",
                null_block_id=0,
            ),
        )
    ]


def test_commit_context_keeps_upstream_for_triton_layers(monkeypatch):
    created, upstream_calls = [], []
    fn = commit_context(monkeypatch, created, upstream_calls)
    builder = commit_builder({"a": NS(), "b": NS(_pla_replay_records=None)})

    assert fn(builder) == "upstream-context"
    assert upstream_calls == [builder]
    assert created == []


def test_commit_context_returns_the_cached_context(monkeypatch):
    created, upstream_calls = [], []
    fn = commit_context(monkeypatch, created, upstream_calls)
    builder = commit_builder({"a": NS(_pla_replay_records="ra")}, context="cached")

    assert fn(builder) == "cached"
    assert created == upstream_calls == []


def test_commit_context_rejects_mixed_record_formats(monkeypatch):
    created, upstream_calls = [], []
    fn = commit_context(monkeypatch, created, upstream_calls)
    builder = commit_builder({"a": NS(_pla_replay_records="ra"), "b": NS()})

    with pytest.raises(ValueError, match="all use PLA replay"):
        fn(builder)
    assert created == upstream_calls == []


def bind_replay(monkeypatch, replay, probes):
    def get_pla_kda_replay():
        probes.append(True)
        return replay

    sdk = ModuleType(SDK)
    sdk.get_pla_kda_replay = get_pla_kda_replay
    monkeypatch.setitem(sys.modules, SDK, sdk)
    return function(
        KDA,
        "_bind_pla_replay",
        {"torch": NS(float32="fp32", bfloat16="bf16"), "logger": LOGGER},
    )


def recoverssm_layer(**overrides):
    layer = NS(
        use_recoverssm=True,
        head_dim=128,
        conv_size=4,
        conv1d=NS(bias=None, weight=NS(dtype="fp32", device="ppu:0")),
        gate_lower_bound=-5.0,
        spec_query_len=4,
        local_num_heads=12,
        recoverssm_records="upstream-records",
        state_dtype=("bf16", "fp32"),
    )
    layer.__dict__.update(overrides)
    layer.get_state_dtype = lambda: layer.state_dtype
    return layer


def replay_sdk(allocated):
    def allocate(*args):
        allocated.append(args)
        return "pla-records"

    return NS(KDAReplayRecords=NS(allocate=allocate))


def replay_config(dtype="bf16"):
    return NS(model_config=NS(dtype=dtype), scheduler_config=NS(max_num_seqs=8))


def test_bind_gives_an_eligible_layer_pla_records_only(monkeypatch):
    allocated = []
    fn = bind_replay(monkeypatch, replay_sdk(allocated), [])
    layer = recoverssm_layer()
    fn(layer, replay_config())

    assert layer._pla_replay_records == "pla-records"
    assert layer.recoverssm_records is None
    assert allocated == [(8, 12, 4, "ppu:0")]


def test_bind_skips_layers_without_recoverssm(monkeypatch):
    probes = []
    fn = bind_replay(monkeypatch, replay_sdk([]), probes)
    layer = recoverssm_layer(use_recoverssm=False)
    fn(layer, replay_config())

    assert layer._pla_replay_records is None
    assert layer.recoverssm_records == "upstream-records"
    assert probes == []


@pytest.mark.parametrize(
    "replay_available,overrides,dtype",
    [
        (False, {}, "bf16"),
        (True, {"head_dim": 64}, "bf16"),
        (True, {"conv_size": 3}, "bf16"),
        (True, {"conv1d": NS(bias="bias", weight=NS(dtype="fp32"))}, "bf16"),
        (True, {"conv1d": NS(bias=None, weight=NS(dtype="bf16"))}, "bf16"),
        (True, {"gate_lower_bound": None}, "bf16"),
        (True, {"spec_query_len": 0}, "bf16"),
        (True, {"spec_query_len": 33}, "bf16"),
        (True, {}, "fp16"),
        (True, {"state_dtype": ("fp32", "fp32")}, "bf16"),
        (True, {"state_dtype": ("bf16", "bf16")}, "bf16"),
    ],
)
def test_bind_keeps_triton_recoverssm_outside_the_pla_contract(
    monkeypatch, replay_available, overrides, dtype
):
    allocated = []
    replay = replay_sdk(allocated) if replay_available else None
    fn = bind_replay(monkeypatch, replay, [])
    layer = recoverssm_layer(**overrides)
    fn(layer, replay_config(dtype))

    assert layer._pla_replay_records is None
    assert layer.recoverssm_records == "upstream-records"
    assert allocated == []


@pytest.fixture
def pla_sdk(modules):
    def load(*, ppu=True, pla=True):
        modules("vllm.platforms", current_platform=NS(is_ppu=lambda: ppu))
        modules("vllm_sail.envs", VLLM_SAIL_USE_PLA=pla)
        return load_patch("vllm_sail/attention/pla_kda.py")

    return load


@pytest.mark.parametrize("ppu,pla", [(False, True), (True, False)])
def test_replay_sdk_is_off_outside_ppu_pla(pla_sdk, modules, ppu, pla):
    modules(REPLAY)
    assert pla_sdk(ppu=ppu, pla=pla).get_pla_kda_replay() is None


def test_replay_sdk_is_optional(pla_sdk, monkeypatch):
    monkeypatch.setitem(sys.modules, REPLAY, None)
    assert pla_sdk().get_pla_kda_replay() is None


def test_replay_sdk_is_the_pla_vllm_namespace(pla_sdk, modules):
    replay = modules(REPLAY)
    assert pla_sdk().get_pla_kda_replay() is replay


def test_commit_context_hands_pla_the_layer_view_it_reads(pla_sdk, modules):
    created = []

    class Context:
        @classmethod
        def create(cls, layers, **kwargs):
            created.append((layers, kwargs))
            return "pla-context"

    modules(REPLAY, KDAReplayCommitContext=Context)
    layer = NS(
        kv_cache=("conv", "state", "workspace"),
        _pla_replay_records="records",
        recoverssm_records=None,
    )
    context = pla_sdk().PLAKDAReplayCommitContext.create(
        [layer],
        spec_query_len=4,
        max_num_reqs=8,
        conv_state_layout="DS",
        null_block_id=0,
    )

    assert context._context == "pla-context"
    [(views, kwargs)] = created
    assert [vars(view) for view in views] == [
        {"kv_cache": ("conv", "state"), "replay_records": "records"}
    ]
    assert kwargs == dict(
        spec_query_len=4, max_num_reqs=8, conv_state_layout="DS", null_block_id=0
    )


def test_commit_trims_the_request_mapping_to_the_batch(pla_sdk):
    commits = []
    context = pla_sdk().PLAKDAReplayCommitContext(
        NS(commit=lambda *args, **kw: commits.append((args, kw)))
    )
    states = NS(shape=(2,))

    assert context.commit("accepted", states, "qsl", [4, 5, 6], "bt", "nct", 64) is None
    context.commit("accepted", states, "qsl")
    context.commit("accepted", NS(shape=(0,)), "qsl", [4])

    assert commits == [
        (
            ("accepted", states, "qsl"),
            dict(
                request_indices=[4, 5],
                block_table="bt",
                num_computed_tokens="nct",
                mamba_block_size=64,
            ),
        ),
        (
            ("accepted", states, "qsl"),
            dict(
                request_indices=None,
                block_table=None,
                num_computed_tokens=None,
                mamba_block_size=None,
            ),
        ),
    ]


def enable_recoverssm(self, config):
    self.use_recoverssm = True
    self.conv_size = 4
    self.conv1d = NS(bias=None, weight=NS(dtype="fp32", device="ppu:0"))
    self.gate_lower_bound = -5.0
    self.spec_query_len = 2
    self.recoverssm_records = "upstream-records"


def test_registered_constructor_binds_pla_replay(kda_constructor, modules):
    allocated = []
    c = kda_constructor(after=enable_recoverssm)
    modules(SDK, get_pla_kda_replay=lambda: replay_sdk(allocated))
    config, cfg = constructor_configs("triton", seqs=4)
    layer = c.provider.KimiK3DeltaAttention(config, cfg, "model.kda", None, False)

    assert layer._pla_replay_records == "pla-records"
    assert layer.recoverssm_records is None
    assert allocated == [(4, 2, 2, "ppu:0")]


def test_registered_replay_patches_delegate_without_pla_records(kda_constructor):
    c = kda_constructor()
    config, cfg = constructor_configs("triton")
    layer = c.provider.KimiK3DeltaAttention(config, cfg, "model.kda", None, False)
    builder_cls = sys.modules[
        "vllm.models.kimi_k3.nvidia.kda_metadata"
    ].KimiK3KDAMetadataBuilder
    builder = builder_cls()
    builder.__dict__.update(vars(commit_builder({"model.kda": layer})))

    assert layer._pla_replay_records is None
    with pytest.raises(AssertionError, match="verify is not part"):
        layer._recoverssm_verify(*(object() for _ in range(7)))
    with pytest.raises(AssertionError, match="commit is not part"):
        builder._get_recoverssm_context()


@pytest.mark.upstream_source
def test_commit_adapter_accepts_upstream_commit_keywords(upstream_source_root):
    adapter = definition(
        ROOT / "vllm_sail/attention/pla_kda.py", "PLAKDAReplayCommitContext.commit"
    )
    upstream = definition(
        upstream_source_root / "vllm/models/kimi_k3/nvidia/ops/recoverssm.py",
        "KDARecoverSSMCommitContext.commit",
    )
    assert [a.arg for a in adapter.args.args] == [a.arg for a in upstream.args.args]
    assert len(adapter.args.defaults) == len(upstream.args.defaults)
