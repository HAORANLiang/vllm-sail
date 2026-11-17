# SPDX-License-Identifier: Apache-2.0
"""The Qwen4Exp QSA indexer's MXFP4 compressed cache route.

Three tiers, following the suite's split:

* Bare CPU: ``ops/qsa_prepare_fp4.py`` is a copied upstream body, and
  ``_qsa_prepare_kernel`` is not a patch target, so ``tools/check_patch_drift.py``
  never hashes it. Collapsing its ``PPU MODIFICATION`` regions must reproduce
  upstream's text exactly; that is the only drift guard the kernel has.
* With torch: the MXFP4 page geometry helpers are pure tensor math and run on
  CPU, as does the DeepGEMM wrapper's call shape against a recorder.
* With torch and vLLM: the patch module's platform delegation and the cache
  spec it publishes.

Device tests cover the Triton packing kernel and the DeepGEMM launches.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PREPARE_FP4 = ROOT / "vllm_sail" / "models" / "qwen4_exp" / "ops" / "qsa_prepare_fp4.py"
PATCH_FILE = (
    ROOT / "vllm_sail" / "patch" / "enhancement" / "models" / "qwen4_exp_qsa_fp4.py"
)
UPSTREAM_PREPARE = "vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py"

_PATCH_MODULE = "vllm_sail.patch.enhancement.models.qwen4_exp_qsa_fp4"


def _definition_lines(path: pathlib.Path, name: str) -> list[str]:
    source = path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(source))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"missing or ambiguous {name!r} in {path}"
    (node,) = matches
    return source[node.lineno - 1 : node.end_lineno]


def _upstream_view(lines: list[str]) -> list[str]:
    """Collapse ``PPU MODIFICATION`` regions back to the text they replaced.

    A region either only adds, or it wraps upstream's statement in the ``else``
    of a PPU-only branch; the ``else`` body, dedented one level, is then exactly
    upstream's text.
    """
    collapsed: list[str] = []
    index = 0
    while index < len(lines):
        if "PPU MODIFICATION: begin" not in lines[index]:
            collapsed.append(lines[index])
            index += 1
            continue
        end = index
        while "PPU MODIFICATION: end" not in lines[end]:
            end += 1
        region = lines[index + 1 : end]
        else_at = next(
            (i for i in reversed(range(len(region))) if region[i].strip() == "else:"),
            None,
        )
        if else_at is not None:
            body = region[else_at + 1 :]
            kept = min(len(line) - len(line.lstrip()) for line in body if line.strip())
            drop = kept - (len(region[else_at]) - len(region[else_at].lstrip()))
            collapsed.extend(line[drop:] if line.strip() else line for line in body)
        index = end + 1
    return collapsed


@pytest.mark.upstream_source
@pytest.mark.parametrize("name", ["_qsa_prepare_kernel", "qsa_prepare"])
def test_copied_prepare_reduces_to_upstream(upstream_source_root, name):
    copied = _upstream_view(_definition_lines(PREPARE_FP4, name))
    assert copied == _definition_lines(upstream_source_root / UPSTREAM_PREPARE, name)


@pytest.fixture(scope="module")
def mxfp4():
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")
    return importlib.import_module("vllm_sail.models.qwen4_exp.ops.mxfp4")


def test_mxfp4_row_geometry(mxfp4):
    assert mxfp4.MXFP4_LOGICAL_HEAD_DIM == 128
    assert mxfp4.MXFP4_PACKED_HEAD_DIM == 64
    # Four 32-dim block scales fit in the int32 the kernels take as ``k_sf``.
    assert mxfp4.MXFP4_SCALE_BYTES == 4
    assert mxfp4.MXFP4_ROW_BYTES == 68


@pytest.mark.parametrize(
    "shape, match",
    [
        ((1, 64, 2, 68), "one kv head"),
        ((1, 64, 1, 128), "68 bytes wide"),
        ((1, 100, 1, 68), "multiple of 64"),
    ],
)
def test_check_qsa_fp4_cache_layout_rejects_a_layout_it_cannot_address(
    mxfp4, shape, match
):
    import torch

    cache = torch.zeros(shape, dtype=torch.uint8)
    with pytest.raises(ValueError, match=match):
        mxfp4.check_qsa_fp4_cache_layout(cache)


@pytest.mark.parametrize("block_rows", [64, 448])
def test_check_qsa_fp4_cache_layout_returns_the_block_as_the_page(mxfp4, block_rows):
    """One framework block is one page, so the page row count is the block's own.

    448 is production's decode geometry: ``--block-size 1792`` at a
    ``compress_ratio`` of 4.
    """
    import torch

    cache = torch.zeros((3, block_rows, 1, 68), dtype=torch.uint8)
    assert mxfp4.check_qsa_fp4_cache_layout(cache) == block_rows


@pytest.mark.parametrize("block_rows", [64, 256])
@pytest.mark.parametrize("bound_view", [False, True], ids=["allocated", "bound"])
@pytest.mark.parametrize(
    "pitch_pad, offset", [(0, 0), (4096, 128)], ids=["tight", "padded"]
)
def test_gather_qsa_fp4_keys_reads_the_planar_block(
    mxfp4, block_rows, bound_view, pitch_pad, offset
):
    """Split planar FP4 blocks back into per-row values and scales.

    ``_qsa_prepare_kernel`` stores each framework block planar -- all of the
    block's value rows, then all of its scale rows -- because DeepGEMM's paged
    kernel reads scales at ``block_base + page_kv * head_dim`` with ``page_kv``
    the block's own row count. Reading 68 contiguous bytes per row instead would
    shift every row past the first and take the scales out of the value plane,
    and a plane per 64 rows would put a 256-row block's scales 12288 bytes too
    early -- both silently, since the shape still validates. ``block_rows=256``
    (``--block-size 1024``) is what distinguishes the two.

    Request lengths scale with ``block_rows`` so both parametrizations walk more
    than one block of their table.

    The compressed cache is one slice of the combined hybrid KV block, so its
    block pitch also spans the other caches: production binds 448 rows with
    ``stride(0)`` in the millions. ``pitch_pad`` covers that, and ``offset`` a
    nonzero ``storage_offset``, both of which the writer shares because it steps
    blocks by ``stride(0)`` from ``data_ptr()``.

    ``bound_view`` reproduces ``bind_kv_cache``'s transpose, whose dim-2 stride is
    the block pitch rather than 0.
    """
    import torch

    packed, row_bytes = mxfp4.MXFP4_PACKED_HEAD_DIM, mxfp4.MXFP4_ROW_BYTES
    num_blocks = 8
    rows = num_blocks * block_rows
    block_bytes = block_rows * row_bytes
    pitch = block_bytes + pitch_pad
    logical = torch.arange(rows, dtype=torch.int32)
    value_bytes = ((logical[:, None] + torch.arange(packed)[None, :]) & 0xFF).to(
        torch.uint8
    )
    scale_bytes = logical.view(torch.uint8).view(rows, mxfp4.MXFP4_SCALE_BYTES)
    # Per block: every value row of the block, then every scale row.
    values = value_bytes.reshape(num_blocks, block_rows * packed)
    scales = scale_bytes.reshape(num_blocks, block_rows * mxfp4.MXFP4_SCALE_BYTES)
    blocks = torch.cat([values, scales], dim=1)

    span = (num_blocks - 1) * pitch + block_bytes
    backing = torch.full((span + offset,), 0xA5, dtype=torch.uint8)
    # Write each block into its own slice of the pitch; a padded span is not
    # ``num_blocks`` blocks end to end, so it cannot be viewed as one matrix.
    backing.as_strided((num_blocks, block_bytes), (pitch, 1), offset).copy_(blocks)
    dim2 = pitch if bound_view else row_bytes
    cache = backing.as_strided(
        (num_blocks, block_rows, 1, row_bytes), (pitch, row_bytes, dim2, 1), offset
    )

    block_table = torch.tensor([[7, 2, 0], [5, 1, 3]], dtype=torch.int32)
    lengths = (block_rows + 6, block_rows - 4)
    cu_seqlens = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int64)
    capacity = int(cu_seqlens[-1])
    k_values, k_sf = mxfp4.gather_qsa_fp4_keys(cache, block_table, cu_seqlens, capacity)

    assert k_values.shape == (capacity, packed) and k_values.dtype == torch.int8
    assert k_sf.shape == (capacity,) and k_sf.dtype == torch.int32
    # Request 0 walks block_table[0] and request 1 walks block_table[1], each from
    # its own row 0, so the expected slot is whichever block the row falls in.
    want_sf = [
        int(block_table[req, row // block_rows]) * block_rows + row % block_rows
        for req, length in enumerate(lengths)
        for row in range(length)
    ]
    assert k_sf.tolist() == want_sf
    for slot, logical_row in enumerate(k_sf.tolist()):
        want = ((logical_row + torch.arange(packed)) & 0xFF).to(torch.uint8)
        assert torch.equal(k_values[slot].view(torch.uint8), want)


@pytest.mark.parametrize(
    "shape",
    [(2, 64, 2, 68), (2, 64, 1, 128)],
)
def test_gather_qsa_fp4_keys_rejects_a_shape_it_cannot_address(mxfp4, shape):
    import torch

    cache = torch.zeros(shape, dtype=torch.uint8)
    block_table = torch.zeros((1, shape[0]), dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match=r"\[blocks, rows, 1, 68\]"):
        mxfp4.gather_qsa_fp4_keys(cache, block_table, cu_seqlens, 1)


def test_gather_qsa_fp4_keys_rejects_a_block_that_is_not_whole_tiles(mxfp4):
    """A block DeepGEMM cannot take as a page has no valid scale plane.

    ``page_kv`` must be a multiple of its 64-row KV compute tile, and the scale
    plane sits behind all of the block's value rows, so a 100-row block would
    place that plane where the kernel reads values -- and
    ``check_qsa_fp4_cache_layout`` rejects the same geometry, so the paged path
    could never read what the gather would write.
    """
    import torch

    cache = torch.zeros((2, 100, 1, mxfp4.MXFP4_ROW_BYTES), dtype=torch.uint8)
    block_table = torch.zeros((1, 2), dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="multiple of 64"):
        mxfp4.gather_qsa_fp4_keys(cache, block_table, cu_seqlens, 1)


def test_gather_qsa_fp4_keys_rejects_a_pitch_smaller_than_a_block(mxfp4):
    """Overlapping blocks cannot be addressed, however plausible the shape looks.

    ``as_strided`` bounds-checks against the storage, so the backing has to be
    large enough for torch to build the bad view before the gather can reject it.
    """
    import torch

    block_rows = 64
    block_bytes = block_rows * mxfp4.MXFP4_ROW_BYTES
    pitch = block_bytes - 8
    cache = torch.zeros(2 * pitch + block_bytes, dtype=torch.uint8).as_strided(
        (2, block_rows, 1, mxfp4.MXFP4_ROW_BYTES),
        (pitch, mxfp4.MXFP4_ROW_BYTES, mxfp4.MXFP4_ROW_BYTES, 1),
    )
    block_table = torch.zeros((1, 2), dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int64)
    with pytest.raises(ValueError, match="block pitch"):
        mxfp4.gather_qsa_fp4_keys(cache, block_table, cu_seqlens, 1)


def test_fp4_avg_logits_call_shape(monkeypatch):
    """The prefill scorer's contract with DeepGEMM, without a device.

    ``clean_logits`` is not a parameter of the FP4 average entry points, so it
    must not be forwarded; ``max_seqlen_k`` is mandatory and must be the caller's
    real bound; and the logits must not come back bf16, which is the FP4 entry
    points' own default.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm", reason="requires vLLM")
    from vllm_sail.utils import deep_gemm

    calls: list[tuple[tuple, dict]] = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return "logits"

    monkeypatch.setattr(deep_gemm, "_fp4_mqa_avg_logits_impl", recorder)
    assert deep_gemm.has_fp4_mqa_avg_logits()

    q, kv, ks, ke = object(), object(), object(), object()
    assert deep_gemm.fp4_mqa_avg_logits(q, kv, ks, ke, 96) == "logits"
    args, kwargs = calls[0]
    assert args == (q, kv, ks, ke)
    assert "clean_logits" not in kwargs
    assert kwargs["max_seqlen_k"] == 96
    assert kwargs["logits_dtype"] is torch.float32
    assert "q_scale" not in kwargs

    calls.clear()
    deep_gemm.fp4_mqa_avg_logits(q, kv, ks, ke, 96, q_scale="scale")
    assert calls[0][1]["q_scale"] == "scale"


def test_fp4_avg_metadata_extra_matches_the_schedule_assert():
    """DeepGEMM sizes the FP4 paged AVG schedule buffer from this exact tuple.

    The head dimension is the logical 128, not the packed 64 the Q/K tensors
    carry: DeepGEMM halves it internally for MXFP4 and rejects a D32 tile. The
    fourth entry is a dtype, since an integer byte size is not accepted.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm", reason="requires vLLM")
    from vllm_sail.utils import deep_gemm

    assert deep_gemm.fp4_avg_metadata_extra(3) == (3, 4, 128, torch.int8)


def test_require_fp4_mqa_avg_logits_needs_both_kernels(monkeypatch):
    pytest.importorskip("vllm", reason="requires vLLM")
    from vllm_sail.models.qwen4_exp.ops import fp4_select
    from vllm_sail.utils import deep_gemm

    monkeypatch.setattr(deep_gemm, "has_fp4_mqa_avg_logits", lambda: True)
    for paged in (False, True):
        monkeypatch.setattr(
            deep_gemm, "has_fp4_paged_mqa_avg_logits", lambda paged=paged: paged
        )
        if paged:
            fp4_select.require_fp4_mqa_avg_logits()
        else:
            with pytest.raises(RuntimeError, match="fp8_fp4_paged_mqa_avg_logits"):
                fp4_select.require_fp4_mqa_avg_logits()


@pytest.fixture(scope="module")
def qsa_fp4_patch():
    """The module object whose globals the installed wrappers actually read.

    ``test_model_patches`` executes every ``models/`` patch file under a
    synthetic module name, so the targets can already be patched while
    ``sys.modules`` has no entry for the real import path. Importing it then
    would trip the double-patch guard; follow the installed wrapper's
    ``__module__`` instead.
    """
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")
    import sys

    from vllm.models.qwen4_exp.nvidia.ops import qsa_prepare as prepare_ops

    from vllm_sail.patch.utils import PATCH_MARKER

    installed = prepare_ops.qsa_prepare
    if getattr(installed, PATCH_MARKER, None) is None:
        return importlib.import_module(_PATCH_MODULE)
    owner = sys.modules.get(getattr(installed, "__module__", ""))
    assert owner is not None, "the installed qsa_prepare wrapper lost its module"
    return owner


def _gate(patch_module, ppu_15: bool):
    """Pin the patch module's generation gate.

    ``is_ppu_15`` is ``functools.cache``d on a static device property, so the
    value this host resolves would leak across tests; binding the module's own
    name keeps each case independent.
    """
    return mock.patch.object(patch_module, "is_ppu_15", return_value=ppu_15)


@pytest.mark.parametrize(
    "wrapper, upstream, fp4",
    [
        ("qsa_prepare", "_upstream_qsa_prepare", "_fp4_qsa_prepare"),
        (
            "qsa_select_paged_decode",
            "_upstream_select_paged_decode",
            "_fp4_select_paged_decode",
        ),
        (
            "qsa_select_paged_prefill",
            "_upstream_select_paged_prefill",
            "_fp4_select_paged_prefill",
        ),
        (
            "warmup_qsa_mqa_paged_decode",
            "_upstream_warmup_paged_decode",
            "_fp4_warmup_paged_decode",
        ),
    ],
)
def test_scoring_wrappers_delegate_on_the_chip_generation(
    qsa_fp4_patch, wrapper, upstream, fp4, monkeypatch
):
    calls: list[str] = []
    for name, label in ((upstream, "upstream"), (fp4, "fp4")):
        monkeypatch.setattr(
            qsa_fp4_patch, name, lambda label=label: calls.append(label)
        )
    body = getattr(qsa_fp4_patch, wrapper)

    # ZW-810E and a non-PPU host are one case to this module: neither has
    # DeepGEMM's FP4 average kernels, so both keep upstream's Triton scorer.
    with _gate(qsa_fp4_patch, False):
        assert body() is None and calls == ["upstream"]
    calls.clear()
    with _gate(qsa_fp4_patch, True):
        assert body() is None and calls == ["fp4"]


def test_wrappers_are_installed_on_every_target(qsa_fp4_patch):
    from vllm.models.qwen4_exp.nvidia import qsa_indexer as indexer_alias
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as ops
    from vllm.models.qwen4_exp.nvidia.ops import qsa_prepare as prepare_ops

    assert prepare_ops.qsa_prepare is qsa_fp4_patch.qsa_prepare
    assert indexer_alias.qsa_prepare is qsa_fp4_patch.qsa_prepare
    assert ops.qsa_select_paged_decode is qsa_fp4_patch.qsa_select_paged_decode
    assert ops.qsa_select_paged_prefill is qsa_fp4_patch.qsa_select_paged_prefill
    assert ops.warmup_qsa_mqa_paged_decode is qsa_fp4_patch.warmup_qsa_mqa_paged_decode


def test_compressed_cache_init_publishes_an_mxfp4_row(qsa_fp4_patch, monkeypatch):
    import torch
    from vllm.models.qwen4_exp.common.qsa_indexer import QSACompressedKeyCache

    calls: list[dict] = []
    probes: list[bool] = []
    monkeypatch.setattr(
        qsa_fp4_patch,
        "_QSAIndexerCache",
        SimpleNamespace(__init__=lambda self, **kwargs: calls.append(kwargs)),
    )
    monkeypatch.setattr(
        qsa_fp4_patch,
        "require_fp4_mqa_avg_logits",
        lambda: probes.append(True),
    )
    init = QSACompressedKeyCache.__init__
    kwargs = {
        "head_size": 128,
        "dtype": torch.bfloat16,
        "cache_config": object(),
        "prefix": "model.layers.0.self_attn.indexer.compressed_k_cache",
        "vllm_config": object(),
    }

    # ZW-810E keeps upstream's cache verbatim, so this patch is a no-op there
    # rather than the model-load failure a bare is_ppu() gate used to cause.
    with _gate(qsa_fp4_patch, False):
        init(object(), **kwargs)
    assert calls[-1]["head_size"] == 128
    assert calls[-1]["dtype"] is torch.bfloat16
    assert calls[-1]["compress_ratio"] == 1
    assert probes == []

    # Off 890P the indexer dtype is upstream's business, not ours.
    with _gate(qsa_fp4_patch, False):
        init(object(), **{**kwargs, "dtype": torch.float8_e4m3fn})
    assert calls[-1]["dtype"] is torch.float8_e4m3fn
    assert probes == []

    with _gate(qsa_fp4_patch, True):
        init(object(), **kwargs, compress_ratio=4)
    assert calls[-1]["head_size"] == qsa_fp4_patch.MXFP4_ROW_BYTES
    assert calls[-1]["dtype"] is torch.uint8
    assert calls[-1]["compress_ratio"] == 4
    assert probes == [True]

    # The FP4 cache has no fp8 fallback: --indexer-kv-dtype fp8 must fail at
    # model load rather than bind a cache no scorer can read.
    with (
        _gate(qsa_fp4_patch, True),
        pytest.raises(NotImplementedError, match="indexer-kv-dtype"),
    ):
        init(object(), **{**kwargs, "dtype": torch.float8_e4m3fn})


def test_kernel_block_sizes_follow_the_fp4_page(qsa_fp4_patch):
    import torch
    from vllm.models.qwen4_exp.common.qsa_indexer import QSAIndexerBackend
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.kv_cache_interface import CircularBufferSpec, MLAAttentionSpec

    block_sizes = QSAIndexerBackend.get_supported_kernel_block_sizes
    compressed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=68,
        dtype=torch.uint8,
        tokens_per_state=4,
    )
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )

    with _gate(qsa_fp4_patch, True):
        # One framework block is one FP4 page and DeepGEMM tiles KV in 64-row
        # compute blocks, so a compress_ratio of 4 pushes the block size to a
        # multiple of 256.
        assert block_sizes(compressed) == [MultipleOf(64 * 4)]
        # The raw key ring is bf16 and never reaches an FP4 kernel.
        assert block_sizes(raw) == [MultipleOf(1)]
        assert block_sizes(None) == [MultipleOf(1)]
    with _gate(qsa_fp4_patch, False):
        assert block_sizes(compressed) == [MultipleOf(1)]


def test_every_patch_point_gates_on_the_same_generation_predicate():
    """The six patch points have to agree on one predicate.

    ``test_capability_matrix.py`` already pins ``is_ppu_15()`` to ZW-M890P; this
    guards the other half. A cache laid out for the MXFP4 scorer and read by
    upstream's Triton one -- or the reverse -- corrupts silently instead of
    raising, so a bare ``is_ppu()`` left at any single site is worse than no
    gate at all.
    """
    source = PATCH_FILE.read_text(encoding="utf-8")
    assert source.count("if not is_ppu_15():") == 6
    assert "current_platform" not in source
