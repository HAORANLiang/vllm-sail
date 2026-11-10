# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric E2E oracles for the PPU FlashMLA stack (CI plan PART 4).

PPU does not build vLLM's ``_flashmla_C`` / ``_flashmla_extension_C``
extensions.  It ships the upstream ``flash_mla`` package and rebinds every
``vllm.v1.attention.ops.flashmla`` entry point to it, which means three
behaviours differ from stock vLLM and all three are asserted here:

* ``FlashMLABackend.supports_compute_capability`` accepts major ``8`` instead
  of ``[9, 10]`` -- without the inversion PPU could never select FlashMLA;
* the FP8 *dense* decode path is routed to the sail kernel, because PPU does
  not build ``torch.ops._flashmla_extension_C``:
  ``get_mla_metadata_dense_fp8`` forwards to the sail ``get_mla_metadata`` (a
  lazy ``FlashMLASchedMeta``, kept off upstream's int32/CUDA-graph path by the
  ``_build_decode`` patch) and ``flash_mla_with_kvcache_fp8`` dispatches through
  ``flash_mla.flash_mla_with_kvcache`` (qkv_fp8), folding ``q_scale * k_scale``
  into ``softmax_scale`` and rescaling the output by ``k_scale``;
* the packed DeepSeek-V4 K cache is decoded by a software E4M3FN Triton kernel
  on 810E (``torch.ops.vllm.ppu_deepseek_v4_dequant_gather``), since Triton
  refuses ``tl.float8e4nv`` on capability (8, 0).

Numeric cases mirror the exact call shape used by
``FlashMLAImpl.forward_mqa`` (dense) and ``FlashMLASparseImpl._bf16_flash_mla_``
``kernel`` (sparse prefill) and compare against the fp32 goldens in
``tests/e2e/kernels/_reference.py`` with the shared ``mla`` band (2e-2).  The
810E packed-cache cases carry ``cap80``; everything else runs on both chips and
carries only ``ppu``.  The ``native_kernels`` fixture asserts the ``flash_mla``
distribution and both custom ops are present, so a missing operator fails the
run instead of skipping it.
"""

from __future__ import annotations

import importlib.util
import re

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU/CUDA device", allow_module_level=True)
pytest.importorskip("vllm")

import vllm_sail  # noqa: E402

vllm_sail.register_out_of_tree()

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_ppu():
    pytest.skip("requires a PPU platform (is_ppu() is False)", allow_module_level=True)

from vllm.platforms.interface import DeviceCapability  # noqa: E402
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402
from vllm.v1.attention.ops import flashmla as mla_ops  # noqa: E402

from tests.e2e.kernels import _reference as ref  # noqa: E402
from vllm_sail.attention.ops import mla_sparse  # noqa: E402
from vllm_sail.platform import _get_backend_priorities  # noqa: E402

pytestmark = pytest.mark.ppu

DEVICE = torch.device("cuda")

_NO_FLASHMLA = (
    "the PPU `flash_mla` distribution is mandatory for this suite: PPU binds "
    "every vllm.v1.attention.ops.flashmla entry point to it, so install the "
    "FlashMLA wheel shipped with the PPU SDK"
)

# DeepSeek-V3/V4 MLA geometry.  The dense kernel consumes a 576-wide packed
# row (512 latent + 64 RoPE) and emits the 512-wide latent, i.e. head_dim_v
# is *not* head_dim.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
MLA_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
#: FlashMLA only implements a 64-token page; see
#: ``FlashMLABackend.get_supported_kernel_block_sizes``.
BLOCK_SIZE = 64
NUM_Q_HEADS = 128
SOFTMAX_SCALE = MLA_HEAD_DIM**-0.5
#: Both FlashMLA entry points report a **natural-log** LSE.  Measured on
#: ZW-890P for the sparse prefill kernel: ``lse`` equals the fp32 natural-log
#: golden element-for-element, while the 2-based rescaling this file used to
#: apply inflated every one of the 8192 entries by exactly ``log2(e)``.

# The fp32 golden materialises ``[B, H, chunk, Sk]``; 64 rows keeps the
# ``H=128, Sk=16384`` decode case at ~32 MiB per chunk.
QUERY_CHUNK = 64

# PART 4: kv_len in {512, 4096, 16384}, q_heads=128, kv_heads=1.  Each case
# uses two requests of unequal length so the tile scheduler and the per-request
# bottom-right causal alignment are both exercised; the last entry adds a
# uniform query length > 1, which is the speculative-decode shape
# ``reshape_query_for_spec_decode`` produces.
DECODE_CASES = [
    pytest.param(1, (512, 384), id="sq1-kv512"),
    pytest.param(1, (4096, 2048), id="sq1-kv4096"),
    pytest.param(1, (16384, 16320), id="sq1-kv16384"),
    pytest.param(4, (512, 300), id="sq4-spec-kv512"),
]

# (num_heads, seq_q, kv_len, topk).  The BF16 sparse prefill kernel requires
# ``h_q`` to be a multiple of 64 and ``topk`` to be a multiple of 128 -- the
# latter is a hard C++ precondition, ``sparse_prefill.h:31`` aborts the process
# with ``params.topk % (2*B_TOPK) == 0`` otherwise, and production honours it by
# sizing the index buffer with ``round_up(top_k + window_size +
# max_image_tokens, 128)`` (``vllm/models/deepseek_v4/nvidia/flashmla.py``).
SPARSE_TOPK_MULTIPLE = 128
SPARSE_CASES = [
    pytest.param(64, 32, 512, 128, id="h64-sq32-kv512-topk128"),
    pytest.param(128, 64, 4096, 128, id="h128-sq64-kv4096-topk128"),
]

# (seq_lens, gather_lens, offset, pad_stride).  ``gather_lens=None`` reproduces
# the compressor's default where the whole sequence is gathered; a non-None
# value exercises the sliding window, ``offset`` the output row shift and
# ``pad=False`` the unpadded block stride.
PACKED_CACHE_CASES = [
    pytest.param((128, 200), None, 0, True, id="full-offset0-padded"),
    pytest.param((65, 129), None, 1, True, id="unaligned-offset1"),
    pytest.param((512, 320), (64, 128), 0, True, id="gather-window"),
    pytest.param((64, 192), None, 0, False, id="full-unpadded-stride"),
]

#: Head width the absorbed/uncompressed equivalence test compresses to.
ABSORBED_NOPE = 128
ABSORBED_KV_LEN = 256


@pytest.fixture(scope="module", autouse=True)
def native_kernels():
    """Load the native extension strictly; a missing wheel must fail."""
    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()

    assert importlib.util.find_spec("flash_mla") is not None, _NO_FLASHMLA
    assert mla_ops._is_flashmla_available() == (True, None), _NO_FLASHMLA
    assert mla_ops.is_flashmla_dense_supported() == (True, None), _NO_FLASHMLA
    assert mla_ops.is_flashmla_sparse_supported() == (True, None), _NO_FLASHMLA
    # A stubbed entry point is the raise-helper itself; the PPU patch must have
    # replaced all three with lazy ``flash_mla`` wrappers.
    for name in ("flash_mla_with_kvcache", "flash_mla_sparse_fwd", "get_mla_metadata"):
        assert getattr(mla_ops, name) is not mla_ops._raise_flashmla_unavailable, (
            f"{_NO_FLASHMLA}: vllm.v1.attention.ops.flashmla.{name} is still the "
            "unavailable stub, so the PPU FlashMLA provider patch did not install"
        )
    # Registered by ``vllm_sail.models.deepseek_v4.ops.cache.register_ops()`` on
    # both chips; only the *routing* is 810E-specific.
    assert hasattr(torch.ops.vllm, "ppu_deepseek_v4_dequant_gather"), (
        "torch.ops.vllm.ppu_deepseek_v4_dequant_gather is missing: the packed MLA "
        "cache gather kernel was not registered by register_out_of_tree()"
    )
    assert hasattr(torch.ops.vllm, "ppu_sparse_attn_indexer"), (
        "torch.ops.vllm.ppu_sparse_attn_indexer is missing: vllm_sail.ops did not "
        "register the sparse indexer"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generator(seed: int = 20240517) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(*shape: int, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, generator=_generator(seed), device=DEVICE, dtype=dtype)


def _backend(member: str) -> type:
    """Resolve ``AttentionBackendEnum.<member>`` to its backend class."""
    return getattr(AttentionBackendEnum, member).get_class()


def _declared_mutations(schema) -> set[str]:
    """Names of the arguments a schema declares mutable.

    ``direct_register_custom_op(mutates_args=[...])`` reaches the schema as an
    alias annotation carrying a write flag, rendered ``Tensor(a!) out``.  The
    accessor for that flag moved between torch releases -- ``is_write`` on older
    builds, ``is_mutable``/``mutable`` on newer ones -- and all of them answer
    the same question, so they are probed in order.  A build exposing none of
    them still renders the annotation, so the canonical schema text is the last
    resort; the asserted contract (which arguments may be written through) is
    identical on every path.
    """
    rendered = None
    mutated = set()
    for argument in schema.arguments:
        alias_info = argument.alias_info
        if alias_info is None:
            continue
        flag = None
        for attribute in ("is_mutable", "mutable", "is_write"):
            flag = getattr(alias_info, attribute, None)
            if flag is not None:
                break
        if flag is None:
            if rendered is None:
                rendered = str(schema)
            # ``Tensor(a!) name`` / ``Tensor(a!)? name=None``: the ``!`` inside
            # the alias annotation is the mutation marker.
            flag = re.search(
                r"\([^()]*![^()]*\)\s*\??\s*" + re.escape(argument.name) + r"\b",
                rendered,
            )
        if flag:
            mutated.add(argument.name)
    return mutated


def _assert_relative_rms(actual, expected, budget: float, msg: str = "") -> None:
    """Keep the shared band discriminating when ``|expected|`` is small.

    With random operands a softmax over ``K`` keys averages the values down to
    roughly ``1 / sqrt(K_eff)``, so an output whose entries are ``~0.1`` is
    already covered by the ``mla`` band's absolute half and the comparison
    would pass for almost any kernel.  This adds a scale-free check: the RMS
    error must stay within ``budget`` of the golden's own RMS.  bf16 storage
    rounding alone contributes ~1e-3, so 5e-2 still fails loudly on a wrong
    softmax scale, a wrong mask or a swapped rope/nope split.
    """
    a = actual.detach().to(torch.float32).cpu()
    e = expected.detach().to(torch.float32).cpu()
    rms = float(e.pow(2).mean().sqrt())
    error = float((a - e).pow(2).mean().sqrt())
    assert error <= budget * rms, (
        f"{msg}: RMS error {error:.3e} exceeds {budget:g} x golden RMS {rms:.3e}"
    )


def _paged_layout(seq_lens, block_size: int, seed: int):
    """Shuffled physical block table plus the matching absolute slot mapping.

    Returns ``(block_table int32 [B, max_blocks], slot_mapping int64 [total],
    num_blocks)``.  One extra block is allocated and left zero-filled so an
    off-by-one in the kernel's block walk reads garbage instead of valid data.
    """
    lengths = [int(length) for length in seq_lens]
    per_request = [(length + block_size - 1) // block_size for length in lengths]
    used = sum(per_request)
    num_blocks = used + 1
    order = torch.randperm(
        num_blocks, generator=_generator(seed), device=DEVICE
    ).tolist()
    spare = order[used]

    rows = []
    slots = []
    cursor = 0
    width = max(per_request)
    for index, length in enumerate(lengths):
        physical = order[cursor : cursor + per_request[index]]
        cursor += per_request[index]
        rows.append(physical + [spare] * (width - len(physical)))
        for position in range(length):
            block = physical[position // block_size]
            slots.append(block * block_size + position % block_size)
    block_table = torch.tensor(rows, dtype=torch.int32, device=DEVICE)
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    return block_table, slot_mapping, num_blocks


def _fill_paged_cache(tokens, slot_mapping, num_blocks, block_size, dtype):
    """Scatter ``[total, width]`` rows into a ``[num_blocks, block_size, width]``."""
    width = tokens.shape[-1]
    cache = torch.zeros((num_blocks, block_size, width), dtype=dtype, device=DEVICE)
    cache.view(-1, width).index_copy_(0, slot_mapping, tokens.to(dtype))
    return cache


def _dense_windows(tokens_per_request, seq_lens):
    """Re-stack flat token rows into ``[B, max_len, width]`` per request.

    Shorter requests are zero-padded to ``max_len``, which is exactly what the
    gather kernel leaves in the untouched tail of ``out``.
    """
    width = tokens_per_request[0].shape[-1]
    longest = max(int(length) for length in seq_lens)
    out = tokens_per_request[0].new_zeros((len(seq_lens), longest, width))
    for index, length in enumerate(seq_lens):
        out[index, : int(length)] = tokens_per_request[index][: int(length)]
    return out


def _run_dense_decode(q, cache, block_table, seq_lens_t, query_len: int):
    """Mirror ``FlashMLAImpl.forward_mqa``'s non-FP8 branch verbatim."""
    scheduler_metadata, _ = mla_ops.get_mla_metadata(
        seq_lens_t,
        query_len * NUM_Q_HEADS,  # num_q_tokens_per_head_k = max_query_len * h_q / h_k
        1,  # MQA for the decode path
        is_fp8_kvcache=False,
    )
    return mla_ops.flash_mla_with_kvcache(
        q=q,
        k_cache=cache.unsqueeze(-2),  # add the single KV head, as upstream does
        block_table=block_table,
        cache_seqlens=seq_lens_t,
        head_dim_v=KV_LORA_RANK,
        tile_scheduler_metadata=scheduler_metadata,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        is_fp8_kvcache=False,
    )


def _run_dense_decode_fp8(
    q, cache_fp8, block_table, seq_lens_t, query_len: int, descale_q, descale_k
):
    """Mirror ``FlashMLAImpl.forward_mqa``'s FP8 branch.

    ``q`` is the real (bf16) query and ``cache_fp8`` the stored FP8 cache; the
    patch quantizes ``q`` with ``descale_q`` internally, exactly as the backend
    hands it the un-quantized query plus the layer scales.
    """
    tile_scheduler_metadata, num_splits = mla_ops.get_mla_metadata_dense_fp8(
        seq_lens_t,
        query_len * NUM_Q_HEADS,  # num_q_tokens_per_head_k = max_query_len * h_q / h_k
        1,  # MQA for the decode path
    )
    return mla_ops.flash_mla_with_kvcache_fp8(
        q=q,
        k_cache=cache_fp8.unsqueeze(-2),  # add the single KV head, as upstream does
        block_table=block_table,
        cache_seqlens=seq_lens_t,
        head_dim_v=KV_LORA_RANK,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
        descale_q=descale_q,
        descale_k=descale_k,
    )


def _sparse_golden(q, kv, indices, lengths, scale):
    """fp32 golden for ``flash_mla_sparse_fwd``.

    Returns ``(output [s_q, H, 512], peak [s_q, H], lse [s_q, H])`` where
    ``lse`` is natural-log and ``peak`` is the largest scaled logit.  Each query
    row attends only to its own valid ``indices`` entries and the selection is
    unordered, so the mask is non-causal.
    """
    seq_q, heads, head_dim = q.shape
    latent = head_dim - QK_ROPE_HEAD_DIM
    qf = q.to(torch.float32)
    kvf = kv.to(torch.float32)
    output = qf.new_zeros((seq_q, heads, latent))
    peak = qf.new_zeros((seq_q, heads))
    lse = qf.new_zeros((seq_q, heads))
    for row in range(seq_q):
        chosen = indices[row, : lengths[row]].to(torch.int64)
        tokens = kvf[chosen]
        kv_c = tokens[:, :latent].unsqueeze(0)
        k_pe = tokens[:, latent:].unsqueeze(0).unsqueeze(2)
        q_row = qf[row].unsqueeze(0).unsqueeze(0)
        output[row] = ref.ref_mla(q_row, kv_c, k_pe, causal=False, scale=scale)[0, 0]
        lse[row] = ref.ref_mla_lse(q_row, kv_c, k_pe, causal=False, scale=scale)[
            0, :, 0
        ]
        scores = (
            torch.einsum("hd,ld->hl", qf[row, :, :latent], kv_c[0])
            + torch.einsum("hd,ld->hl", qf[row, :, latent:], k_pe[0, :, 0])
        ) * scale
        peak[row] = scores.amax(dim=-1)
    return output, peak, lse


# ---------------------------------------------------------------------------
# Backend contract: the degradation chain PART 4 asks for
# ---------------------------------------------------------------------------


def test_mla_backend_priority_chain_is_flashmla_then_triton_then_sparse():
    """PPU's MLA preference order, and TokenspeedMLA is never a candidate.

    ``TokenspeedMLABackend.supports_compute_capability`` requires major 10, so
    it is a stub on both PPU chips; if it ever entered the chain it would only
    produce a confusing "no valid backend" diagnostic.
    """
    expected = [
        AttentionBackendEnum.FLASHMLA,
        AttentionBackendEnum.TRITON_MLA,
        AttentionBackendEnum.FLASHMLA_SPARSE,
    ]
    for capability in (DeviceCapability(8, 0), DeviceCapability(8, 9)):
        assert _get_backend_priorities(True, capability, NUM_Q_HEADS) == expected
    assert AttentionBackendEnum.TOKENSPEED_MLA not in expected
    # The non-MLA chain is the FlashAttention one covered by
    # test_flash_attn_numeric.py; assert it is disjoint so a regression that
    # swaps the two lists is caught here too.
    assert AttentionBackendEnum.FLASHMLA not in _get_backend_priorities(
        False, DeviceCapability(8, 0), NUM_Q_HEADS
    )


def test_flashmla_capability_support_is_inverted_for_ppu():
    """Upstream accepts majors ``[9, 10]``; PPU accepts major ``8`` only."""
    backend = _backend("FLASHMLA")
    assert backend.supports_compute_capability(DeviceCapability(8, 0))
    assert backend.supports_compute_capability(DeviceCapability(8, 9))
    assert not backend.supports_compute_capability(DeviceCapability(9, 0))
    assert not backend.supports_compute_capability(DeviceCapability(10, 0))


def test_mla_backend_capability_matrix():
    """Sparse FlashMLA keeps Hopper/Blackwell, Triton is unconditional."""
    sparse = _backend("FLASHMLA_SPARSE")
    triton = _backend("TRITON_MLA")
    tokenspeed = _backend("TOKENSPEED_MLA")
    for capability in (
        DeviceCapability(8, 0),
        DeviceCapability(8, 9),
        DeviceCapability(9, 0),
        DeviceCapability(10, 0),
    ):
        assert sparse.supports_compute_capability(capability), capability
        assert triton.supports_compute_capability(capability), capability
    assert not tokenspeed.supports_compute_capability(DeviceCapability(8, 0))
    assert not tokenspeed.supports_compute_capability(DeviceCapability(8, 9))
    assert tokenspeed.supports_compute_capability(DeviceCapability(10, 0))


def test_flashmla_declares_a_64_token_page_and_bf16_operands():
    """The geometry this suite hard-codes is the geometry the backend declares."""
    backend = _backend("FLASHMLA")
    assert backend.get_supported_kernel_block_sizes() == [BLOCK_SIZE]
    assert backend.supported_dtypes == [torch.float16, torch.bfloat16]
    assert "auto" in backend.supported_kv_cache_dtypes
    assert "fp8_e4m3" in backend.supported_kv_cache_dtypes
    assert backend.get_name() == "FLASHMLA"
    assert _backend("FLASHMLA_SPARSE").get_name() == "FLASHMLA_SPARSE"
    assert _backend("TRITON_MLA").get_name() == "TRITON_MLA"


def test_flashmla_availability_probe_follows_the_wheel():
    """``_is_flashmla_available`` looks for ``flash_mla``, not ``_flashmla_C``.

    This is the single patch that decides whether PPU gets FlashMLA at all, so
    it is pinned against the distribution the fixture already proved present.
    """
    available, reason = mla_ops._is_flashmla_available()
    assert available is True
    assert reason is None


def test_fp8_dense_metadata_returns_sched_meta_on_ppu():
    """``get_mla_metadata_dense_fp8`` forwards to the sail ``get_mla_metadata``.

    PPU does not build ``_flashmla_extension_C``; the sail kernel fills its
    scheduling metadata lazily from a ``FlashMLASchedMeta``, so the patch returns
    ``(FlashMLASchedMeta, None)`` — the same shape the non-FP8 path uses — and
    the ``_build_decode`` patch keeps upstream's int32/CUDA-graph copies off PPU.
    """
    from flash_mla import FlashMLASchedMeta

    seq_lens = torch.tensor([128, 64], dtype=torch.int32, device=DEVICE)
    sched, splits = mla_ops.get_mla_metadata_dense_fp8(seq_lens, NUM_Q_HEADS, 1)
    assert isinstance(sched, FlashMLASchedMeta)
    assert not sched.have_initialized
    assert splits is None


@pytest.mark.cap89
@pytest.mark.parametrize(("query_len", "seq_lens"), DECODE_CASES)
def test_fp8_dense_flashmla_matches_golden(query_len, seq_lens):
    """PPU FP8 dense decode routes to the sail qkv_fp8 kernel (was a hard error).

    The sail kernel exposes no descale parameters, so the patch quantizes ``q``
    with ``descale_q``, folds ``q_scale * k_scale`` into ``softmax_scale`` (QK
    part) and rescales the output by ``k_scale`` (V part). The golden runs on
    the dequantized operands, so the residual is only the kernel's own FP8
    compute, bounded by the ``mla_fp8_cache`` band. Requires sm89 (the sail FP8
    dense kernel rejects older capability).
    """
    batch = len(seq_lens)
    total = sum(int(length) for length in seq_lens)
    q_scale, k_scale = 0.088, 0.064

    q = _randn(batch, query_len, NUM_Q_HEADS, MLA_HEAD_DIM, seed=20250101)
    tokens = _randn(total, MLA_HEAD_DIM, seed=20250102)
    block_table, slot_mapping, num_blocks = _paged_layout(seq_lens, BLOCK_SIZE, seed=7)
    cache = _fill_paged_cache(
        tokens, slot_mapping, num_blocks, BLOCK_SIZE, torch.bfloat16
    )
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)

    # Quantize q and the cache to fp8_e4m3 with the calibrated scales; the
    # dequantized operands are exactly what the kernel logically attends over.
    q_fp8 = (q.to(torch.float32) / q_scale).to(torch.float8_e4m3fn)
    cache_fp8 = (cache.to(torch.float32) / k_scale).to(torch.float8_e4m3fn)
    q_dq = q_fp8.to(torch.float32) * q_scale
    cache_dq = cache_fp8.to(torch.float32) * k_scale

    descale_q = torch.tensor([q_scale], dtype=torch.float32, device=DEVICE)
    descale_k = torch.tensor([k_scale], dtype=torch.float32, device=DEVICE)

    output, lse = _run_dense_decode_fp8(
        q, cache_fp8, block_table, seq_lens_t, query_len, descale_q, descale_k
    )

    assert output.shape == (batch, query_len, NUM_Q_HEADS, KV_LORA_RANK)
    assert output.dtype == torch.bfloat16
    assert lse.shape == (batch, NUM_Q_HEADS, query_len)
    assert lse.dtype == torch.float32

    golden = ref.ref_mla_paged(
        q_dq,
        cache_dq,
        block_table,
        seq_lens_t,
        latent=KV_LORA_RANK,
        causal=True,
        scale=SOFTMAX_SCALE,
        query_chunk=QUERY_CHUNK,
    )
    ref.assert_close(
        output, golden, "mla_fp8_cache", msg=f"fp8 dense decode {seq_lens}"
    )


def test_degradation_reason_reaches_the_backend_selector(monkeypatch):
    """Without the wheel ``supports_combination`` must report why and fall back.

    Follows the ``fork_port`` monkeypatch pattern: force the probe to the
    unavailable state and assert the reason string travels through
    ``FlashMLABackend.supports_combination``, which is what
    ``PPUPlatform.get_valid_backends`` records as the invalid-backend reason
    before trying TRITON_MLA.
    """
    reason = "ppu flashmla is not available, please install flashmla. "
    monkeypatch.setattr(mla_ops, "is_flashmla_dense_supported", lambda: (False, reason))
    monkeypatch.setattr(
        mla_ops, "is_flashmla_sparse_supported", lambda: (False, reason)
    )
    backend = _backend("FLASHMLA")
    kwargs = {
        "head_size": MLA_HEAD_DIM,
        "dtype": torch.bfloat16,
        "kv_cache_dtype": "auto",
        "block_size": BLOCK_SIZE,
        "use_mla": True,
        "has_sink": False,
        "use_mm_prefix": False,
        "device_capability": DeviceCapability(8, 0),
    }
    assert backend.supports_combination(use_sparse=False, **kwargs) == reason
    assert backend.supports_combination(use_sparse=True, **kwargs) == reason

    # The capability gate still passes, so the reason above -- not the
    # capability -- is what removes FLASHMLA from the candidate list.
    assert backend.supports_compute_capability(DeviceCapability(8, 0))
    assert (
        _get_backend_priorities(True, DeviceCapability(8, 0))[0]
        == AttentionBackendEnum.FLASHMLA
    )


def test_deepseek_v4_flashmla_keeps_the_native_q_head_count():
    """PPU does not round ``h_q`` up to the FP8 kernel's {64, 128}.

    Upstream returns ``64 if num_heads <= 64 else 128`` because its FP8 decode
    kernel only implements those two widths; PPU routes FP8 dense decode to the
    sail kernel, which does not pad heads, so the override returns ``num_heads``
    unchanged and still rejects ``> 128``.
    """
    from vllm_sail.models.deepseek_v4.flashmla import (
        DeepseekV4FlashMLAAttention as attention,
    )

    assert attention.get_padded_num_q_heads(NUM_Q_HEADS) == NUM_Q_HEADS
    assert attention.get_padded_num_q_heads(64) == 64
    # 96 would be padded to 128 upstream; PPU keeps it, which is the observable
    # difference the override exists for.
    assert attention.get_padded_num_q_heads(96) == 96
    with pytest.raises(ValueError, match="does not support 256 heads"):
        attention.get_padded_num_q_heads(256)


# ---------------------------------------------------------------------------
# Dense FlashMLA decode numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("query_len", "seq_lens"), DECODE_CASES)
def test_flashmla_decode_matches_golden(query_len, seq_lens):
    """``flash_mla_with_kvcache`` over a shuffled paged 576-wide cache."""
    batch = len(seq_lens)
    total = sum(int(length) for length in seq_lens)

    q = _randn(batch, query_len, NUM_Q_HEADS, MLA_HEAD_DIM, seed=20250101)
    tokens = _randn(total, MLA_HEAD_DIM, seed=20250102)
    block_table, slot_mapping, num_blocks = _paged_layout(seq_lens, BLOCK_SIZE, seed=7)
    cache = _fill_paged_cache(
        tokens, slot_mapping, num_blocks, BLOCK_SIZE, torch.bfloat16
    )
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)

    output, lse = _run_dense_decode(q, cache, block_table, seq_lens_t, query_len)

    assert output.shape == (batch, query_len, NUM_Q_HEADS, KV_LORA_RANK)
    assert output.dtype == torch.bfloat16
    assert lse.shape == (batch, NUM_Q_HEADS, query_len)
    assert lse.dtype == torch.float32

    golden = ref.ref_mla_paged(
        q,
        cache,
        block_table,
        seq_lens_t,
        latent=KV_LORA_RANK,
        causal=True,
        scale=SOFTMAX_SCALE,
        query_chunk=QUERY_CHUNK,
    )
    ref.assert_close(output, golden, "mla", msg=f"dense decode {seq_lens}")
    _assert_relative_rms(output, golden, 5e-2, f"dense decode {seq_lens}")

    golden_lse = ref.ref_mla_paged_lse(
        q,
        cache,
        block_table,
        seq_lens_t,
        latent=KV_LORA_RANK,
        causal=True,
        scale=SOFTMAX_SCALE,
        query_chunk=QUERY_CHUNK,
    )
    ref.assert_close(lse, golden_lse, "fp32", atol=1e-2, rtol=1e-2)


def test_flashmla_decode_only_reads_the_declared_sequence_length():
    """Keys past ``cache_seqlens`` live in allocated blocks and must be ignored.

    The block table spans more physical blocks than the sequence uses and the
    tail of the last block is filled with large values, so a kernel that walks
    ``block_table.shape[-1] * block_size`` rows instead of ``cache_seqlens``
    diverges immediately.
    """
    seq_lens = (192, 96)
    total = sum(seq_lens)
    query_len = 1
    q = _randn(len(seq_lens), query_len, NUM_Q_HEADS, MLA_HEAD_DIM, seed=20250103)
    tokens = _randn(total, MLA_HEAD_DIM, seed=20250104)
    block_table, slot_mapping, num_blocks = _paged_layout(seq_lens, BLOCK_SIZE, seed=11)
    cache = _fill_paged_cache(
        tokens, slot_mapping, num_blocks, BLOCK_SIZE, torch.bfloat16
    )
    # Poison every slot the block table can reach but the sequences cannot.
    reachable = block_table.to(torch.int64).reshape(-1) * BLOCK_SIZE
    poison = torch.full((MLA_HEAD_DIM,), 1e3, dtype=torch.bfloat16, device=DEVICE)
    written = set(slot_mapping.tolist())
    flat = cache.view(-1, MLA_HEAD_DIM)
    for block in reachable.tolist():
        for position in range(BLOCK_SIZE):
            slot = block + position
            if slot not in written:
                flat[slot] = poison

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    output, _ = _run_dense_decode(q, cache, block_table, seq_lens_t, query_len)
    golden = ref.ref_mla_paged(
        q,
        cache,
        block_table,
        seq_lens_t,
        latent=KV_LORA_RANK,
        causal=True,
        scale=SOFTMAX_SCALE,
        query_chunk=QUERY_CHUNK,
    )
    assert bool(torch.isfinite(output.to(torch.float32)).all())
    ref.assert_close(output, golden, "mla")
    _assert_relative_rms(output, golden, 5e-2, "cache_seqlens window")


def test_flashmla_absorbed_matches_uncompressed_attention():
    """The ``7168 -> 512`` compression is lossless for the attention score.

    PART 4 lists ``head_dim in {512, 576, 7168 -> compressed}``.  7168 is the
    model hidden size, not a FlashMLA head width: what the kernel actually
    proves is that attending with the *absorbed* 576/512 operands reproduces
    plain 192/128 attention obtained by expanding the latent back to per-head
    K/V.  With ``W_UQ``, ``W_UK``, ``W_UV`` of shape ``[512, 128]`` per head:

    ``q_abs = W_UK (W_UQ^T q_lat)`` gives ``q_abs . c == (W_UQ^T q) . (W_UK^T c)``
    and ``W_UV^T o_latent == sum_k p_k v_k``, so the absorbed 512-wide output
    projected by ``W_UV`` must equal the uncompressed 128-wide output.
    """
    batch, query_len, kv_len = 1, 1, ABSORBED_KV_LEN
    heads, nope = NUM_Q_HEADS, ABSORBED_NOPE
    fan_in = KV_LORA_RANK**-0.5

    q_lat = _randn(
        batch, query_len, heads, KV_LORA_RANK, seed=3001, dtype=torch.float32
    )
    q_rope = _randn(batch, query_len, heads, QK_ROPE_HEAD_DIM, seed=3002)
    # Scale the latent up so the softmax is peaked and the output is large
    # enough for the 2e-2 band to stay meaningful.
    c = _randn(batch, kv_len, KV_LORA_RANK, seed=3003, dtype=torch.float32).mul_(2.0)
    k_pe = _randn(batch, kv_len, QK_ROPE_HEAD_DIM, seed=3004)
    w_uq = _randn(heads, KV_LORA_RANK, nope, seed=3005, dtype=torch.float32).mul_(
        fan_in
    )
    w_uk = _randn(heads, KV_LORA_RANK, nope, seed=3006, dtype=torch.float32).mul_(
        fan_in
    )
    w_uv = _randn(heads, KV_LORA_RANK, nope, seed=3007, dtype=torch.float32).mul_(
        fan_in
    )

    # --- uncompressed (per-head) operands, all fp32 -------------------------
    q_nope = torch.einsum("bshe,hef->bshf", q_lat, w_uq)
    k_nope = torch.einsum("bke,hef->bkhf", c, w_uk)
    v = torch.einsum("bke,hef->bkhf", c, w_uv)

    # --- absorbed (FlashMLA) operands ---------------------------------------
    # The compression identity itself, before any kernel is involved:
    # ``(W_UK W_UQ^T q) . c == (W_UQ^T q) . (W_UK^T c)``.
    q_abs_nope = torch.einsum("bshf,hef->bshe", q_nope, w_uk)
    plain = torch.einsum("bshf,bkhf->bshk", q_nope, k_nope)
    absorbed = torch.einsum("bshe,bke->bshk", q_abs_nope, c)
    ref.assert_close(
        absorbed,
        plain,
        "fp32",
        atol=1e-2,
        rtol=1e-2,
        msg="absorbed and per-head scores must agree before quantisation",
    )

    q_abs = torch.cat([q_abs_nope.to(torch.bfloat16), q_rope], dim=-1)
    kv = torch.cat([c.to(torch.bfloat16), k_pe], dim=-1)
    block_table, slot_mapping, num_blocks = _paged_layout(
        (kv_len,), BLOCK_SIZE, seed=13
    )
    cache = _fill_paged_cache(
        kv.reshape(-1, MLA_HEAD_DIM),
        slot_mapping,
        num_blocks,
        BLOCK_SIZE,
        torch.bfloat16,
    )
    seq_lens_t = torch.tensor([kv_len], dtype=torch.int32, device=DEVICE)
    o_latent, _ = _run_dense_decode(q_abs, cache, block_table, seq_lens_t, query_len)
    absorbed_output = torch.einsum("bshe,hef->bshf", o_latent.to(torch.float32), w_uv)

    # --- golden: plain attention on the expanded 192-wide operands ----------
    k_full = torch.cat(
        [
            k_nope,
            k_pe.to(torch.float32)
            .unsqueeze(2)
            .expand(batch, kv_len, heads, QK_ROPE_HEAD_DIM),
        ],
        dim=-1,
    )
    q_full = torch.cat([q_nope, q_rope.to(torch.float32)], dim=-1)
    expected = ref.ref_attention(
        q_full.permute(0, 2, 1, 3).to(torch.bfloat16),
        k_full.permute(0, 2, 1, 3).to(torch.bfloat16),
        v.permute(0, 2, 1, 3).to(torch.bfloat16),
        causal=True,
        scale=SOFTMAX_SCALE,
        query_chunk=QUERY_CHUNK,
    ).permute(0, 2, 1, 3)
    assert expected.shape == absorbed_output.shape
    ref.assert_close(
        absorbed_output,
        expected,
        "mla",
        msg="FlashMLA's absorbed 576/512 path must equal uncompressed 192/128",
    )
    _assert_relative_rms(absorbed_output, expected, 5e-2, "absorbed vs uncompressed")

    # The absorbed latent output on its own must still match the MLA golden.
    golden_latent = ref.ref_mla_paged(
        q_abs,
        cache,
        block_table,
        seq_lens_t,
        latent=KV_LORA_RANK,
        causal=True,
        scale=SOFTMAX_SCALE,
        query_chunk=QUERY_CHUNK,
    )
    ref.assert_close(o_latent, golden_latent, "mla")


# ---------------------------------------------------------------------------
# Sparse FlashMLA prefill numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("num_heads", "seq_q", "kv_len", "topk"), SPARSE_CASES)
def test_flashmla_sparse_prefill_matches_golden(num_heads, seq_q, kv_len, topk):
    """``flash_mla_sparse_fwd`` with ragged, partially invalid selections.

    Mirrors ``FlashMLASparseImpl._bf16_flash_mla_kernel``: the cache is viewed
    as ``[total, 1, 576]`` and the indices as ``[s_q, 1, topk]``, with the
    unused tail of each row set to ``-1`` and its true length passed through
    ``topk_length``.
    """
    q = _randn(seq_q, num_heads, MLA_HEAD_DIM, seed=4001)
    kv = _randn(kv_len, MLA_HEAD_DIM, seed=4002)

    # Ragged selection lengths so ``topk_length`` is not a no-op, and every row
    # keeps at least one valid index (an all-``-1`` row is documented as having
    # undefined out/lse).
    lengths = [topk - (row % 4) for row in range(seq_q)]
    indices = torch.full((seq_q, topk), -1, dtype=torch.int32, device=DEVICE)
    generator = _generator(4003)
    for row in range(seq_q):
        chosen = torch.randperm(kv_len, generator=generator, device=DEVICE)
        indices[row, : lengths[row]] = chosen[: lengths[row]].to(torch.int32)
    topk_length = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)

    output, max_logits, lse = mla_ops.flash_mla_sparse_fwd(
        q,
        kv.view(-1, 1, MLA_HEAD_DIM),
        indices.view(seq_q, 1, -1),
        SOFTMAX_SCALE,
        topk_length=topk_length,
    )

    assert output.shape == (seq_q, num_heads, KV_LORA_RANK)
    assert output.dtype == torch.bfloat16
    assert max_logits.shape == (seq_q, num_heads)
    assert lse.shape == (seq_q, num_heads)
    assert lse.dtype == torch.float32

    golden, golden_peak, golden_lse = _sparse_golden(
        q, kv, indices, lengths, SOFTMAX_SCALE
    )
    ref.assert_close(output, golden, "mla", msg=f"sparse prefill h={num_heads}")
    _assert_relative_rms(output, golden, 5e-2, f"sparse prefill h={num_heads}")
    # max_logits is a maximum of raw scaled scores, so it carries no log base.
    ref.assert_close(max_logits, golden_peak, "fp32", atol=1e-2, rtol=1e-2)
    # The documented LSE is natural-log, same as the golden.
    ref.assert_close(lse, golden_lse, "fp32", atol=1e-2, rtol=1e-2)


def test_flashmla_sparse_prefill_ignores_out_of_range_indices():
    """Indices ``>= s_kv`` are the second documented invalid sentinel.

    The kernel contract accepts either ``-1`` or a value at or beyond the cache
    length.  Rows here mix both sentinels with valid entries, so a kernel that
    only masks ``-1`` would read out of bounds.
    """
    num_heads, seq_q, kv_len, topk = 64, 8, 256, SPARSE_TOPK_MULTIPLE
    q = _randn(seq_q, num_heads, MLA_HEAD_DIM, seed=4004)
    kv = _randn(kv_len, MLA_HEAD_DIM, seed=4005)

    lengths = [topk - 4] * seq_q
    indices = torch.full((seq_q, topk), -1, dtype=torch.int32, device=DEVICE)
    generator = _generator(4006)
    for row in range(seq_q):
        chosen = torch.randperm(kv_len, generator=generator, device=DEVICE)
        valid = chosen[: lengths[row]].to(torch.int32)
        indices[row, : lengths[row]] = valid
        # Half the rows mark the padding with an out-of-range index instead.
        if row % 2:
            indices[row, lengths[row] :] = kv_len + row
    topk_length = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)

    output, _, lse = mla_ops.flash_mla_sparse_fwd(
        q,
        kv.view(-1, 1, MLA_HEAD_DIM),
        indices.view(seq_q, 1, -1),
        SOFTMAX_SCALE,
        topk_length=topk_length,
    )
    golden, _, golden_lse = _sparse_golden(q, kv, indices, lengths, SOFTMAX_SCALE)
    assert bool(torch.isfinite(output.to(torch.float32)).all())
    assert bool(torch.isfinite(lse).all())
    ref.assert_close(output, golden, "mla")
    _assert_relative_rms(output, golden, 5e-2, "out-of-range sparse indices")
    ref.assert_close(lse, golden_lse, "fp32", atol=1e-2, rtol=1e-2)


#: Absolute bound on how far ``lse`` may move between two calls with identical
#: inputs.  Measured on ZW-890P over four shapes (sq8/kv256 ragged,
#: sq16/kv512 dense at topk 128 and 256, sq32/kv512 ragged) with three repeats
#: each: ``output`` and ``max_logits`` were bitwise identical every single time,
#: while 2-3% of the ``lse`` entries moved by at most **2 ulp** (9.54e-07
#: absolute, on values near 5.2, i.e. ~1.8e-7 relative).  That is the
#: log-sum-exp accumulator summing its per-tile partials in a schedule-dependent
#: order -- benign float non-associativity, recorded as an SDK observation.
#: 1e-5 keeps >10x head-room over the measurement while a genuinely flaky
#: split-K reduction would blow through it by orders of magnitude and would move
#: ``output`` with it.
LSE_REPEATABILITY_ATOL = 1e-5


def test_flashmla_sparse_prefill_is_bitwise_repeatable():
    """Same inputs, same bytes: a flaky split-K reduction must not pass.

    ``output`` and ``max_logits`` are held to bitwise equality, which the kernel
    meets exactly.  ``lse`` is held to :data:`LSE_REPEATABILITY_ATOL` for the
    reason documented there.
    """
    num_heads, seq_q, kv_len, topk = 64, 16, 512, SPARSE_TOPK_MULTIPLE
    q = _randn(seq_q, num_heads, MLA_HEAD_DIM, seed=4007)
    kv = _randn(kv_len, MLA_HEAD_DIM, seed=4008)
    lengths = [topk] * seq_q
    indices = torch.empty((seq_q, topk), dtype=torch.int32, device=DEVICE)
    generator = _generator(4009)
    for row in range(seq_q):
        chosen = torch.randperm(kv_len, generator=generator, device=DEVICE)
        indices[row] = chosen[:topk].to(torch.int32)
    topk_length = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)

    arguments = (
        q,
        kv.view(-1, 1, MLA_HEAD_DIM),
        indices.view(seq_q, 1, -1),
        SOFTMAX_SCALE,
    )
    first = mla_ops.flash_mla_sparse_fwd(*arguments, topk_length=topk_length)
    second = mla_ops.flash_mla_sparse_fwd(*arguments, topk_length=topk_length)
    assert torch.equal(first[0], second[0]), "sparse prefill output is not reproducible"
    assert torch.equal(first[1], second[1]), "max_logits is not reproducible"
    lse_drift = (first[2] - second[2]).abs().max().item()
    assert lse_drift <= LSE_REPEATABILITY_ATOL, (
        f"lse moved by {lse_drift!r} between two calls with identical inputs, "
        f"above the {LSE_REPEATABILITY_ATOL!r} bound measured for the kernel's "
        "non-associative log-sum-exp accumulator"
    )


# ---------------------------------------------------------------------------
# Packed MLA cache: 810E software E4M3FN decode/gather
# ---------------------------------------------------------------------------


@pytest.mark.cap80
@pytest.mark.parametrize(
    ("seq_lens", "gather_lens", "offset", "pad"), PACKED_CACHE_CASES
)
def test_packed_mla_cache_gather_matches_golden(seq_lens, gather_lens, offset, pad):
    """``torch.ops.vllm.ppu_deepseek_v4_dequant_gather`` byte-for-byte layout.

    Triton cannot compile ``tl.float8e4nv`` on capability (8, 0), so 810E
    decodes the packed cache's E4M3FN bytes in software.  The cache is built by
    the reference encoder (the shipped encoder is fused into the compressor and
    cannot be driven standalone), then decoded twice: once through the patched
    upstream entry point and once through the custom op directly.
    """
    from vllm.models.deepseek_v4.common.ops import cache_utils

    requests = len(seq_lens)
    total = sum(int(length) for length in seq_lens)
    k = _randn(total, ref.PACKED_MLA_HEAD_DIM, seed=5001)
    block_table, slot_mapping, num_blocks = _paged_layout(seq_lens, BLOCK_SIZE, seed=17)
    k_cache = ref.build_packed_mla_cache(
        k, slot_mapping, num_blocks, BLOCK_SIZE, pad=pad
    )
    assert k_cache.dtype == torch.uint8
    assert k_cache.shape == (
        num_blocks,
        ref.packed_mla_block_stride(BLOCK_SIZE, pad=pad),
    )

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    gather_lens_t = (
        None
        if gather_lens is None
        else torch.tensor(gather_lens, dtype=torch.int32, device=DEVICE)
    )
    effective = seq_lens if gather_lens is None else gather_lens
    max_rows = offset + max(int(length) for length in effective)
    out = torch.zeros(
        (requests, max_rows, ref.PACKED_MLA_HEAD_DIM),
        dtype=torch.bfloat16,
        device=DEVICE,
    )

    cache_utils.dequantize_and_gather_k_cache(
        out,
        k_cache,
        seq_lens_t,
        gather_lens_t,
        block_table,
        BLOCK_SIZE,
        offset,
    )

    golden = ref.ref_packed_mla_gather(
        k_cache, seq_lens_t, gather_lens_t, block_table, BLOCK_SIZE, offset, max_rows
    )
    ref.assert_close(out, golden, "packed_mla_cache", msg=f"packed gather {seq_lens}")

    # Rows the gather window does not cover must stay at their zero init.
    if offset:
        assert not out[:, :offset].abs().any(), "rows before `offset` were written"
    if gather_lens is not None:
        for request, length in enumerate(gather_lens):
            tail = offset + int(length)
            assert not out[request, tail:].abs().any(), f"row {request} overran"


@pytest.mark.cap80
@pytest.mark.parametrize(
    ("seq_lens", "gather_lens", "offset", "pad"), PACKED_CACHE_CASES[:2]
)
def test_packed_mla_cache_routes_to_the_ppu_op(seq_lens, gather_lens, offset, pad):
    """The patched entry point must dispatch to the PPU op, not to upstream.

    ``patch/enhancement/models/deepseek_v4_cache.py`` only takes the SM80 branch
    when ``is_ppu() and is_device_capability((8, 0))``; calling the custom op
    directly on the same buffers and demanding identical bytes proves the branch
    was taken rather than the upstream Triton kernel (which cannot even compile
    here).
    """
    from vllm.models.deepseek_v4.common.ops import cache_utils

    from vllm_sail.patch.utils import PATCH_MARKER

    markers = getattr(cache_utils.dequantize_and_gather_k_cache, PATCH_MARKER, {})
    assert markers, (
        "cache_utils.dequantize_and_gather_k_cache carries no PPU patch marker: "
        "register_out_of_tree() did not install the packed-cache route"
    )

    total = sum(int(length) for length in seq_lens)
    k = _randn(total, ref.PACKED_MLA_HEAD_DIM, seed=5002)
    block_table, slot_mapping, num_blocks = _paged_layout(seq_lens, BLOCK_SIZE, seed=19)
    k_cache = ref.build_packed_mla_cache(
        k, slot_mapping, num_blocks, BLOCK_SIZE, pad=pad
    )
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    gather_lens_t = (
        None
        if gather_lens is None
        else torch.tensor(gather_lens, dtype=torch.int32, device=DEVICE)
    )
    max_rows = offset + max(int(length) for length in seq_lens)
    shape = (len(seq_lens), max_rows, ref.PACKED_MLA_HEAD_DIM)
    # Both buffers start from the same sentinel so the rows outside the gather
    # window compare equal, and any write at all is observable.
    patched = torch.full(shape, 0.5, dtype=torch.bfloat16, device=DEVICE)
    direct = torch.full(shape, 0.5, dtype=torch.bfloat16, device=DEVICE)

    cache_utils.dequantize_and_gather_k_cache(
        patched, k_cache, seq_lens_t, gather_lens_t, block_table, BLOCK_SIZE, offset
    )
    torch.ops.vllm.ppu_deepseek_v4_dequant_gather(
        direct, k_cache, seq_lens_t, gather_lens_t, block_table, BLOCK_SIZE, offset
    )
    assert torch.equal(patched, direct), (
        "the patched entry point did not produce the PPU SM80 kernel's bytes"
    )
    assert bool((patched != 0.5).any()), "the gather kernel wrote nothing"


@pytest.mark.cap80
def test_packed_mla_cache_rope_half_round_trips_exactly():
    """Only the 448 latent entries are E4M3FN; the 64 RoPE entries are bf16.

    The kernel copies the RoPE half verbatim, so it must be bitwise identical to
    what was stored, while the latent half only has to land inside the relaxed
    E4M3FN band.  This is what separates a working software decoder from one
    that silently mis-scales a quant block.
    """
    from vllm.models.deepseek_v4.common.ops import cache_utils

    seq_lens = (256, 128)
    total = sum(seq_lens)
    k = _randn(total, ref.PACKED_MLA_HEAD_DIM, seed=5003)
    block_table, slot_mapping, num_blocks = _paged_layout(seq_lens, BLOCK_SIZE, seed=23)
    k_cache = ref.build_packed_mla_cache(k, slot_mapping, num_blocks, BLOCK_SIZE)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    max_rows = max(seq_lens)
    out = torch.zeros(
        (len(seq_lens), max_rows, ref.PACKED_MLA_HEAD_DIM),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    cache_utils.dequantize_and_gather_k_cache(
        out, k_cache, seq_lens_t, None, block_table, BLOCK_SIZE, 0
    )

    dense = _dense_windows(list(torch.split(k, list(seq_lens))), seq_lens)
    rope = dense[..., ref.PACKED_MLA_FP8_DIM :]
    assert torch.equal(out[..., ref.PACKED_MLA_FP8_DIM :], rope.to(torch.bfloat16))

    latent = dense[..., : ref.PACKED_MLA_FP8_DIM]
    ref.assert_close(
        out[..., : ref.PACKED_MLA_FP8_DIM],
        latent,
        "mla_fp8_cache",
        "cap80",
        msg="E4M3FN latent decode is outside the relaxed FP8 band",
    )


def test_packed_mla_cache_op_mutates_only_its_out_argument():
    """Schema contract: ``out`` is the single mutated argument.

    ``direct_register_custom_op(mutates_args=["out"])`` is what lets the op sit
    inside a piecewise CUDA graph; a schema drift would break capture silently.
    """
    schema = torch.ops.vllm.ppu_deepseek_v4_dequant_gather.default._schema
    mutated = _declared_mutations(schema)
    assert mutated == {"out"}
    assert [argument.name for argument in schema.arguments] == [
        "out",
        "k_cache",
        "seq_lens",
        "gather_lens",
        "block_table",
        "block_size",
        "offset",
    ]


# ---------------------------------------------------------------------------
# Sparse indexer workspace contracts (vllm_sail/attention/ops/mla_sparse.py)
# ---------------------------------------------------------------------------


def test_kv_cache_as_quant_view_matches_the_documented_layout():
    """FP8 unsqueezes; MXFP4 re-strides into packed nibbles + ue8m0 scales."""
    head_dim = MLA_HEAD_DIM
    num_blocks, block_size = 3, BLOCK_SIZE
    fp8_dtype = current_platform.fp8_dtype()

    fp8_cache = torch.zeros(
        (num_blocks, block_size, head_dim + 4), dtype=fp8_dtype, device=DEVICE
    )
    fp8_view = mla_sparse.kv_cache_as_quant_view(fp8_cache, head_dim, False)
    assert fp8_view.shape == (num_blocks, block_size, 1, head_dim + 4)
    assert fp8_view.data_ptr() == fp8_cache.data_ptr()

    fp4_bytes = head_dim // 2 + head_dim // mla_sparse.MXFP4_BLOCK_SIZE
    fp4_cache = torch.zeros(
        (num_blocks, block_size, fp4_bytes), dtype=torch.uint8, device=DEVICE
    )
    fp4_view = mla_sparse.kv_cache_as_quant_view(fp4_cache, head_dim, True)
    assert fp4_view.shape == (num_blocks, block_size, 1, fp4_bytes)
    assert fp4_view.stride() == (block_size * fp4_bytes, fp4_bytes, fp4_bytes, 1)
    assert fp4_view.data_ptr() == fp4_cache.data_ptr()


def test_gather_workspace_shapes_follow_the_cache_dtype():
    """FP8 keeps one fp32 scale per token; MXFP4 keeps one ue8m0 per 32 values."""
    total, head_dim = 512, MLA_HEAD_DIM
    fp8_dtype = current_platform.fp8_dtype()

    fp8_values, fp8_scales = mla_sparse._gather_workspace_shapes(
        total, head_dim, fp8_dtype, False
    )
    assert fp8_values == ((total, head_dim), fp8_dtype)
    assert fp8_scales == ((total, 4), torch.uint8)

    fp4_values, fp4_scales = mla_sparse._gather_workspace_shapes(
        total, head_dim, fp8_dtype, True
    )
    assert fp4_values == ((total, head_dim // 2), torch.uint8)
    assert fp4_scales == ((total, head_dim // mla_sparse.MXFP4_BLOCK_SIZE), torch.uint8)
    assert mla_sparse.MXFP4_BLOCK_SIZE == 32
    assert mla_sparse.RADIX_TOPK_WORKSPACE_SIZE == 1024 * 1024


def test_sparse_indexer_fake_returns_the_caller_supplied_buffer():
    """The meta implementation is the identity on ``topk_indices_buffer``.

    ``ppu_sparse_attn_indexer`` calls it during the profiling run to reserve
    workspace, so returning anything else would make torch.compile allocate a
    second buffer and desynchronise the real kernel's writes.
    """
    buffer = torch.full((16, 64), -1, dtype=torch.int32, device=DEVICE)
    returned = mla_sparse.ppu_sparse_attn_indexer_fake(
        hidden_states=torch.zeros((16, 7168), dtype=torch.bfloat16, device=DEVICE),
        k_cache_prefix="kv_cache",
        kv_cache=torch.zeros((4, BLOCK_SIZE, 580), dtype=torch.uint8, device=DEVICE),
        q_quant=torch.zeros((16, MLA_HEAD_DIM), dtype=torch.uint8, device=DEVICE),
        q_scale=None,
        k=None,
        weights=torch.zeros((NUM_Q_HEADS, MLA_HEAD_DIM), device=DEVICE),
        quant_block_size=128,
        scale_fmt=None,
        topk_tokens=64,
        head_dim=MLA_HEAD_DIM,
        max_model_len=4096,
        total_seq_lens=16 * 128,
        topk_indices_buffer=buffer,
        skip_k_cache_insert=True,
    )
    assert returned is buffer


def test_sparse_indexer_op_declares_both_mutated_buffers():
    """``topk_indices_buffer`` and ``candidate_blocks`` are the mutated args."""
    schema = torch.ops.vllm.ppu_sparse_attn_indexer.default._schema
    mutated = _declared_mutations(schema)
    assert mutated == {"topk_indices_buffer", "candidate_blocks"}
