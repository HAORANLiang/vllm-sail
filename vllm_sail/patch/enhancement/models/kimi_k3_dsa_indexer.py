# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
# Copied Triton bodies resolve ``tl``/``triton`` in *this* module: a
# ``triton.jit`` function captures its defining module's globals, so
# ``bind_body`` cannot re-point them at the upstream module.
"""810E-class target support for the Kimi K3.1 DSA indexer.

Two hardware gaps on capability ``(8, 0)`` PPU targets (e.g. 810E) break the
stock DSA indexer, which otherwise only needs SM89+:

* Triton's CUDA backend rejects ``tl.float8e4nv`` below capability 8.9 and the
  target has no f32->e4m3 cvt instruction, so the fused prologue's FP8 stores
  fail to compile. The replacement wrappers quantize through a software RNE
  e4m3fn encoder instead (bit-identical to the hardware cast).
* Targets without FP8 MMA also have no FP8 DeepGEMM ``*_mqa_logits`` kernel.
  The wrappers then quantize the indexer Q/K to int8 (ue8m0 scales folded into
  the folded weights, mirroring the DeepSeek V3 PPU int8-indexer gate) and
  route prefill scoring through ``int8_mqa_logits`` and decode/paged scoring
  through ``int8_paged_mqa_logits``, both from ``vllm_sail.utils.deep_gemm``.

Only the three entry points are replaced, with minimal diffs: the kernel
signatures and inner call sites stay byte-identical to upstream (the fork's
SW_FP8/USE_INT8/INT8_MAX knobs ride along as defaulted ``tl.constexpr``
parameters), so every launch reuses the upstream text except the quant
branch and the raw-byte cache views. The stock ``@triton.jit`` kernel objects
stay in place because ``dsa_indexer_mxfp4`` calls
``_dsa_fused_prologue_kernel`` with the old signature; targets with FP8
support delegate to upstream.

The same commit also branches ``vllm.model_executor.layers.sparse_attn_indexer``
on the int8 Q dtype; rather than copying that ~650-line function, its two
module-level scoring helpers are rebound here (see ``_install_int8_scoring``).
Kimi's dense indexer calls that module-level op directly
(``models/kimi_k3/nvidia/dsa.py``), so it bypasses ``ppu_sparse_attn_indexer``
-- the op that already dispatches on int8 Q in
``vllm_sail/attention/ops/mla_sparse.py``.

Known limit: an explicit ``indexer_kv_dtype="mxfp4"`` cache takes
``dsa_indexer_mxfp4``, which this module does not cover; it needs FP8 MMA
anyway, so it never applies below capability 8.9.
"""

from __future__ import annotations

import functools
import importlib

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

import vllm.model_executor.layers.sparse_attn_indexer as _sparse
import vllm.models.kimi_k3.nvidia.ops.dsa_indexer as _dsa_indexer
from vllm.v1.attention.ops.common import pack_ue8m0_x4
from vllm_sail.patch.enhancement.dsa_indexer_gate import (
    int8_indexer_active,
    uses_software_e4m3,
)
from vllm_sail.patch.utils import PATCH_MARKER, patch
from vllm_sail.utils import deep_gemm as _sail_deep_gemm

_AFFECTED = ">=0.34.0,<0.35.0"
_DSA_MODULE = "vllm.models.kimi_k3.nvidia.ops.dsa_indexer"
_SPARSE_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"

logger = init_logger(__name__)

# Same gates as the fork, taken from ``dsa_indexer_gate`` so the attention
# layer's logits-dtype pin reaches the same verdict: when the int8 path is
# downgraded (an older DeepGEMM without the int8 kernels), that pin is the live
# FP8 path and must stay installed.
_SW_FP8_CAST = uses_software_e4m3()
_USE_INT8_INDEXER = int8_indexer_active()
_FP8_DTYPE = current_platform.fp8_dtype()
_USE_FNUZ = torch.float8_e4m3fnuz == _FP8_DTYPE
_FP8_MAX = 224.0 if _USE_FNUZ else 448.0
_INT8_MAX = 127.0

# The software encoder emits E4M3FN bytes only, and an e4b8 (fnuz) cache would
# decode them as garbage; the combination fails loudly instead of writing it.
_SW_ENCODER_ELIGIBLE = not (_SW_FP8_CAST and _USE_FNUZ)


def _require_supported_encoder() -> None:
    """Reject the software encoder on targets that quantize to e4m3fnuz."""
    if _SW_ENCODER_ELIGIBLE:
        return
    raise NotImplementedError(
        "This target needs the software encoder (below capability 8.9) but "
        "quantizes to float8_e4m3fnuz, which this patch does not model; only "
        "the e4m3fn bit layout is implemented."
    )


# Captured before patching so non-affected targets delegate to upstream.
_upstream_k_insert = _dsa_indexer.indexer_k_norm_quant_insert
_upstream_q_quant = _dsa_indexer.indexer_q_quant_fold_weights
_upstream_prologue = _dsa_indexer.dsa_fused_prologue

# Unchanged upstream jit helpers, reused in place: a jit call resolves through
# the callee object itself, so cross-module references are safe.
_ue8m0_scale = _dsa_indexer._ue8m0_scale
_fill_topk = _dsa_indexer._fill_topk
_rms_norm_row = _dsa_indexer._rms_norm_row


# ---------------------------------------------------------------------------
# Fork-only quant helpers (the whole point of the patch)
# ---------------------------------------------------------------------------


@triton.jit
def _encode_e4m3fn(x):
    """Encode pre-clamped fp32 to FP8 E4M3FN as uint8 (SM80 compatible).

    Bit-identical to ``x.to(tl.float8e4nv)`` (RNE, saturating) but uses only
    integer ops, so it compiles where Triton refuses the fp8 dtype.
    """
    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 31) & 1
    abs_bits = bits & 0x7FFFFFFF

    fp32_exp = (abs_bits >> 23) & 0xFF
    fp32_mant = abs_bits & 0x7FFFFF

    is_zero = abs_bits == 0
    fp8_exp_normal = fp32_exp - 120
    is_subnorm = (fp8_exp_normal <= 0) & (~is_zero)

    # ---- Normal path: top 3 mantissa bits with RNE rounding. ----
    truncated = fp32_mant & 0xFFFFF
    halfway = 0x80000
    lsb = (fp32_mant >> 20) & 1
    round_up_n = (truncated > halfway) | ((truncated == halfway) & (lsb == 1))
    normal_mant = (fp32_mant >> 20) + round_up_n.to(tl.int32)
    mant_ovf = normal_mant >= 8
    normal_exp_out = fp8_exp_normal + mant_ovf.to(tl.int32)
    normal_mant_out = tl.where(mant_ovf, 0, normal_mant)

    # ---- Subnormal path: shift the implicit-1 mantissa into the fp8 denormal. ----
    full_mant = 0x800000 | fp32_mant
    sub_shift = 141 - fp32_exp
    sub_shift_safe = tl.minimum(tl.maximum(sub_shift, 1), 31)
    one_i32 = tl.full((), 1, tl.int32)
    round_pos = sub_shift_safe - 1
    round_mask = one_i32 << round_pos
    sticky_mask = round_mask - 1
    sub_round_bit = (full_mant & round_mask) != 0
    sub_sticky = (full_mant & sticky_mask) != 0
    sub_truncated = full_mant >> sub_shift_safe
    sub_lsb = sub_truncated & 1
    sub_round_up = sub_round_bit & (sub_sticky | (sub_lsb == 1))
    sub_mant_out = sub_truncated + sub_round_up.to(tl.int32)
    sub_carry_to_normal = sub_mant_out >= 8
    sub_exp_final = tl.where(sub_carry_to_normal, 1, 0)
    sub_mant_final = tl.where(sub_carry_to_normal, 0, sub_mant_out & 0x7)

    out_exp = tl.where(is_subnorm, sub_exp_final, normal_exp_out)
    out_mant = tl.where(is_subnorm, sub_mant_final, normal_mant_out)
    out_exp = tl.where(is_zero, 0, out_exp)
    out_mant = tl.where(is_zero, 0, out_mant)
    out_exp = tl.minimum(tl.maximum(out_exp, 0), 15)
    out_mant = out_mant & 0x7

    return ((sign << 7) | (out_exp << 3) | out_mant).to(tl.uint8)


@triton.jit
def _quant_to_fp8(
    y,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr = _USE_FNUZ,
    SW_FP8: tl.constexpr = _SW_FP8_CAST,
):
    """Scaled fp32 -> fp8 elements, saturating at +-FP8_MAX. On SW_FP8
    targets the result is e4m3fn *bytes* (uint8) instead of an fp8 value."""
    y = tl.minimum(tl.maximum(y, -FP8_MAX), FP8_MAX)
    if SW_FP8:
        return _encode_e4m3fn(y)
    fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv
    return y.to(fp8_dtype)


@triton.jit
def _quant_to_int8(y, INT8_MAX: tl.constexpr = _INT8_MAX):
    """Scaled fp32 -> nearest int8, saturating at +-INT8_MAX.

    Round-to-nearest-even, matching ``torch.round`` (and the RNE encoder above);
    round-half-up is off by one LSB on every exact tie.
    """
    y = tl.minimum(tl.maximum(y, -INT8_MAX), INT8_MAX)
    floored = tl.math.floor(y)
    as_int = floored.to(tl.int32)
    # Ties round to the even neighbour; ``as_int & 1`` works for negatives in
    # two's complement (-2 & 1 == 0, -3 & 1 == 1).
    round_up = ((y - floored) > 0.5) | (((y - floored) == 0.5) & ((as_int & 1) != 0))
    return (as_int + round_up.to(tl.int32)).to(tl.int8)


# ---------------------------------------------------------------------------
# Kernels: upstream signatures/bodies kept verbatim; only the quant branches
# differ (constexpr-defaulted knobs, so call sites match upstream exactly).
# ---------------------------------------------------------------------------


@triton.jit
def _index_k_insert(
    tok,
    k_ptr,
    norm_w_ptr,
    norm_b_ptr,
    norm_eps,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_scale_ptr,
    cache_block_size,
    cache_block_stride,
    HEAD_DIM: tl.constexpr,
    MX_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr = _USE_FNUZ,
    USE_INT8: tl.constexpr = _USE_INT8_INDEXER,
    INT8_MAX: tl.constexpr = _INT8_MAX,
):
    # Padded rows and (under DCP) tokens owned by another rank have no slot.
    slot = tl.load(slot_mapping_ptr + tok, mask=tok < num_slots, other=-1)
    if slot >= 0:
        offs = tl.arange(0, HEAD_DIM)
        x = tl.load(k_ptr + offs).to(tl.float32)
        mean = tl.sum(x, axis=0) / HEAD_DIM
        diff = x - mean
        rstd = tl.rsqrt(tl.sum(diff * diff, axis=0) / HEAD_DIM + norm_eps)
        w = tl.load(norm_w_ptr + offs).to(tl.float32)
        b = tl.load(norm_b_ptr + offs).to(tl.float32)
        # Round to bf16 like the unfused path so the cached bytes match.
        y = (diff * rstd * w + b).to(k_ptr.dtype.element_ty).to(tl.float32)

        # Same math as the CUDA indexer_k_quant_and_cache kernel.
        # PPU MODIFICATION: int8 quant (ue8m0 scale, raw byte cache) when the
        # target has no FP8 MMA; SW e4m3 encoder below capability 8.9.
        if USE_INT8:
            scale = _ue8m0_scale(tl.max(tl.abs(y), axis=0), 1e-4, INT8_MAX)
            qv = _quant_to_int8(tl.div_rn(y, scale), INT8_MAX)
            qv = qv.to(tl.uint8, bitcast=True)
        else:
            scale = _ue8m0_scale(tl.max(tl.abs(y), axis=0), 1e-4, FP8_MAX)
            qv = _quant_to_fp8(tl.div_rn(y, scale), FP8_MAX, USE_FNUZ)

        # Block layout: [block_size x HEAD_DIM fp8 values | block_size x 4B scale].
        block_start = (slot // cache_block_size) * cache_block_stride
        block_off = slot % cache_block_size
        tl.store(cache_ptr + block_start + block_off * HEAD_DIM + offs, qv)
        scale_byte = block_start + cache_block_size * HEAD_DIM + block_off * 4
        if MX_SCALE:
            scale = pack_ue8m0_x4(scale)
        tl.store(cache_scale_ptr + scale_byte // 4, scale)


@triton.jit
def _indexer_k_norm_quant_insert_kernel(
    k_ptr,
    k_stride,
    norm_w_ptr,
    norm_b_ptr,
    norm_eps,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_scale_ptr,
    cache_block_size,
    cache_block_stride,
    topk_ptr,
    topk_stride,
    HEAD_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    MX_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr = _USE_FNUZ,
):
    tok = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 1:
        # Padding rows included.
        _fill_topk(topk_ptr + tok * topk_stride, TOPK, TOPK_BLOCK)
    else:
        _index_k_insert(
            tok,
            k_ptr + tok * k_stride,
            norm_w_ptr,
            norm_b_ptr,
            norm_eps,
            slot_mapping_ptr,
            num_slots,
            cache_ptr,
            cache_scale_ptr,
            cache_block_size,
            cache_block_stride,
            HEAD_DIM,
            MX_SCALE,
            FP8_MAX,
            USE_FNUZ,
        )


@triton.jit
def _indexer_q_quant_fold_weights_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    w_ptr,
    w_stride,
    q_fp8_ptr,
    q_fp8_stride0,
    q_fp8_stride1,
    w_out_ptr,
    w_out_stride,
    weight_scale,
    HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr = _USE_FNUZ,
    USE_INT8: tl.constexpr = _USE_INT8_INDEXER,
    INT8_MAX: tl.constexpr = _INT8_MAX,
):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + tok * q_stride0 + head * q_stride1 + offs).to(tl.float32)
    # Same math as per_token_group_quant_fp8(use_ue8m0=True) (eps=1e-10).
    # PPU MODIFICATION: int8 quant when the target has no FP8 MMA.
    if USE_INT8:
        scale = _ue8m0_scale(tl.max(tl.abs(q), axis=0), 1e-10, INT8_MAX)
        q_fp8 = _quant_to_int8(tl.div_rn(q, scale), INT8_MAX)
    else:
        scale = _ue8m0_scale(tl.max(tl.abs(q), axis=0), 1e-10, FP8_MAX)
        q_fp8 = _quant_to_fp8(tl.div_rn(q, scale), FP8_MAX, USE_FNUZ)
    tl.store(q_fp8_ptr + tok * q_fp8_stride0 + head * q_fp8_stride1 + offs, q_fp8)
    w = tl.load(w_ptr + tok * w_stride + head).to(tl.float32)
    tl.store(w_out_ptr + tok * w_out_stride + head, w * scale * weight_scale)


@triton.jit
def _dsa_fused_prologue_kernel(
    q_c_ptr,
    q_c_stride,
    q_norm_w_ptr,
    q_out_ptr,
    kv_c_ptr,
    kv_c_stride,
    kv_norm_w_ptr,
    kv_out_ptr,
    k_pe_ptr,
    k_pe_stride,
    index_k_ptr,
    index_k_stride,
    k_norm_w_ptr,
    k_norm_b_ptr,
    eps,
    mla_slot_ptr,
    num_mla_slots,
    mla_cache_ptr,
    mla_scale_ptr,
    mla_rope_ptr,
    mla_block_size,
    mla_block_stride,
    mla_entry_stride,
    index_slot_ptr,
    num_index_slots,
    index_cache_ptr,
    index_scale_ptr,
    index_block_size,
    index_block_stride,
    topk_ptr,
    topk_stride,
    Q_DIM: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    KV_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    INDEX_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    MX_SCALE: tl.constexpr,
    USE_PDL: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr = _USE_FNUZ,
):
    tok = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    if pid == 0:
        _index_k_insert(
            tok,
            index_k_ptr + tok * index_k_stride,
            k_norm_w_ptr,
            k_norm_b_ptr,
            eps,
            index_slot_ptr,
            num_index_slots,
            index_cache_ptr,
            index_scale_ptr,
            index_block_size,
            index_block_stride,
            INDEX_DIM,
            MX_SCALE,
            FP8_MAX,
            USE_FNUZ,
        )
    elif pid == 1:
        # The normed row is written for every token (the prefill path reads it).
        offs = tl.arange(0, KV_DIM)
        y = _rms_norm_row(
            kv_c_ptr + tok * kv_c_stride, kv_norm_w_ptr, eps, KV_DIM, KV_DIM
        )
        y = y.to(kv_out_ptr.dtype.element_ty)
        tl.store(kv_out_ptr + tok * KV_DIM + offs, y)
        slot = tl.load(mla_slot_ptr + tok, mask=tok < num_mla_slots, other=-1)
        if slot >= 0:
            # fp8_ds_mla, same math as concat_and_cache_ds_mla: KV_DIM fp8 values
            # with power-of-two per-128 tile scales (fp32), then k_pe in bf16.
            base = (slot // mla_block_size) * mla_block_stride + (
                slot % mla_block_size
            ) * mla_entry_stride
            NUM_TILES: tl.constexpr = KV_DIM // 128
            tiles = tl.reshape(y.to(tl.float32), (NUM_TILES, 128))
            amax = tl.max(tl.abs(tiles), axis=1, keep_dims=True)
            scale = tl.math.exp2(
                tl.math.ceil(tl.math.log2(tl.maximum(amax / FP8_MAX, 1e-4)))
            )
            # PPU MODIFICATION: SW e4m3 encoder below capability 8.9.
            tl.store(
                mla_cache_ptr + base + offs,
                tl.reshape(_quant_to_fp8(tiles / scale, FP8_MAX, USE_FNUZ), (KV_DIM,)),
            )
            tl.store(
                mla_scale_ptr + base // 4 + KV_DIM // 4 + tl.arange(0, NUM_TILES),
                tl.reshape(scale, (NUM_TILES,)),
            )
            rope = tl.arange(0, ROPE_DIM)
            k_pe = tl.load(k_pe_ptr + tok * k_pe_stride + rope)
            tl.store(
                mla_rope_ptr + base // 2 + KV_DIM // 2 + 2 * NUM_TILES + rope, k_pe
            )
    elif pid == 2:
        offs = tl.arange(0, Q_BLOCK)
        y = _rms_norm_row(q_c_ptr + tok * q_c_stride, q_norm_w_ptr, eps, Q_DIM, Q_BLOCK)
        tl.store(
            q_out_ptr + tok * Q_DIM + offs,
            y.to(q_out_ptr.dtype.element_ty),
            mask=offs < Q_DIM,
        )
    else:
        # Padding rows included.
        _fill_topk(topk_ptr + tok * topk_stride, TOPK, TOPK_BLOCK)


# ---------------------------------------------------------------------------
# Replacement entry points: verbatim upstream launches; only the cache dtype
# / Q allocation lines differ (marked PPU MODIFICATION).
# ---------------------------------------------------------------------------


@patch(
    _DSA_MODULE,
    "indexer_k_norm_quant_insert",
    reason=(
        "Capability (8,0) targets (810E) cannot compile tl.float8e4nv, and "
        "targets without FP8 MMA need an int8 indexer cache, so the K "
        "LayerNorm+quant+insert launch must pass the cache as raw uint8. "
        "constexpr-gated; targets with FP8 support delegate to upstream."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream dsa_indexer selects the quant encoder through a platform hook.",
)
@functools.wraps(_upstream_k_insert)
def indexer_k_norm_quant_insert(
    k,
    norm_weight,
    norm_bias,
    norm_eps,
    slot_mapping,
    kv_cache,
    topk_indices_buffer,
    num_tokens,
):
    if not (_SW_FP8_CAST or _USE_INT8_INDEXER):
        return _upstream_k_insert(
            k,
            norm_weight,
            norm_bias,
            norm_eps,
            slot_mapping,
            kv_cache,
            topk_indices_buffer,
            num_tokens,
        )
    _require_supported_encoder()
    # Verbatim upstream body; only the marked line differs.
    if num_tokens == 0:
        return
    head_dim = k.shape[-1]
    assert k.stride(-1) == 1
    assert kv_cache.dim() == 3 and kv_cache.shape[-1] == head_dim + 4
    assert triton.next_power_of_2(head_dim) == head_dim
    mx_scale = _dsa_indexer.mqa_logits_use_mx_scales()
    u8_cache = kv_cache.view(torch.uint8)
    topk = topk_indices_buffer.shape[-1]
    _indexer_k_norm_quant_insert_kernel[(num_tokens, 2)](
        k,
        k.stride(0),
        norm_weight,
        norm_bias,
        norm_eps,
        slot_mapping,
        slot_mapping.shape[0],
        u8_cache,  # PPU MODIFICATION: raw bytes (SW encoder / int8), no fp8 view
        u8_cache.view(torch.int32 if mx_scale else torch.float32),
        u8_cache.shape[1],
        u8_cache.stride(0),
        topk_indices_buffer,
        topk_indices_buffer.stride(0),
        HEAD_DIM=head_dim,
        TOPK=topk,
        TOPK_BLOCK=min(1024, triton.next_power_of_2(topk)),
        MX_SCALE=mx_scale,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        num_warps=1,
    )


@patch(
    _DSA_MODULE,
    "indexer_q_quant_fold_weights",
    reason=(
        "Capability (8,0) targets (810E) cannot compile tl.float8e4nv, and "
        "targets without FP8 MMA must quantize the indexer Q to int8 (the "
        "downstream sparse_attn_indexer scoring expects it). constexpr-gated; "
        "targets with FP8 support delegate to upstream."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream dsa_indexer selects the quant encoder through a platform hook.",
)
@functools.wraps(_upstream_q_quant)
def indexer_q_quant_fold_weights(
    index_q,
    index_weights,
    weight_scale,
    padded_num_heads,
):
    if not (_SW_FP8_CAST or _USE_INT8_INDEXER):
        return _upstream_q_quant(index_q, index_weights, weight_scale, padded_num_heads)
    _require_supported_encoder()
    # Verbatim upstream body; only the marked lines differ.
    num_tokens, num_heads, head_dim = index_q.shape
    assert index_q.stride(-1) == 1
    assert index_weights.shape == (num_tokens, num_heads)
    assert triton.next_power_of_2(head_dim) == head_dim
    alloc = torch.empty if padded_num_heads == num_heads else torch.zeros
    q_fp8 = alloc(
        (num_tokens, padded_num_heads, head_dim),
        # PPU MODIFICATION: int8 Q when the target has no FP8 MMA.
        dtype=torch.int8 if _USE_INT8_INDEXER else _FP8_DTYPE,
        device=index_q.device,
    )
    weights = alloc(
        (num_tokens, padded_num_heads), dtype=torch.float32, device=index_q.device
    )
    if num_tokens == 0:
        return q_fp8, weights
    _indexer_q_quant_fold_weights_kernel[(num_tokens, num_heads)](
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_weights,
        index_weights.stride(0),
        # PPU MODIFICATION: the SW encoder writes raw e4m3fn bytes.
        q_fp8 if (_USE_INT8_INDEXER or not _SW_FP8_CAST) else q_fp8.view(torch.uint8),
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weight_scale,
        HEAD_DIM=head_dim,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        num_warps=1,
    )
    return q_fp8, weights


@patch(
    _DSA_MODULE,
    "dsa_fused_prologue",
    reason=(
        "Capability (8,0) targets (810E) cannot compile tl.float8e4nv, so the "
        "fused prologue's fp8_ds_mla and indexer-key inserts must quantize "
        "through the software e4m3 encoder (int8 for the key when FP8 MMA is "
        "missing) over raw uint8 cache views. constexpr-gated; targets with "
        "FP8 support delegate to upstream."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream dsa_indexer selects the quant encoder through a platform hook.",
)
@functools.wraps(_upstream_prologue)
def dsa_fused_prologue(
    q_c,
    kv_c,
    k_pe,
    index_k,
    q_norm_weight,
    kv_norm_weight,
    k_norm_weight,
    k_norm_bias,
    eps,
    topk_indices_buffer,
    mla_cache,
    mla_slot_mapping,
    index_cache,
    index_slot_mapping,
):
    if not (_SW_FP8_CAST or _USE_INT8_INDEXER):
        return _upstream_prologue(
            q_c,
            kv_c,
            k_pe,
            index_k,
            q_norm_weight,
            kv_norm_weight,
            k_norm_weight,
            k_norm_bias,
            eps,
            topk_indices_buffer,
            mla_cache,
            mla_slot_mapping,
            index_cache,
            index_slot_mapping,
        )
    _require_supported_encoder()
    # Verbatim upstream body; only the marked lines differ.
    num_tokens, q_dim = q_c.shape
    kv_dim, rope_dim, index_dim = kv_c.shape[1], k_pe.shape[1], index_k.shape[1]
    assert all(t.stride(-1) == 1 for t in (q_c, kv_c, k_pe, index_k))
    q_out = torch.empty((num_tokens, q_dim), dtype=q_c.dtype, device=q_c.device)
    kv_out = torch.empty((num_tokens, kv_dim), dtype=kv_c.dtype, device=kv_c.device)
    if num_tokens == 0:
        return q_out, kv_out

    # No cache (profile runs): an empty slot mapping disables the insert.
    def placeholder() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(1, 1, 4, dtype=torch.uint8, device=q_c.device),
            torch.empty(0, dtype=torch.int64, device=q_c.device),
        )

    if mla_cache is None or mla_slot_mapping is None:
        mla_cache, mla_slot_mapping = placeholder()
    if index_cache is None or index_slot_mapping is None:
        index_cache, index_slot_mapping = placeholder()
    mla = mla_cache.view(torch.uint8)
    index = index_cache.view(torch.uint8)
    mx_scale = _dsa_indexer.mqa_logits_use_mx_scales()
    topk = topk_indices_buffer.shape[-1]
    use_pdl = current_platform.is_arch_support_pdl()
    _dsa_fused_prologue_kernel[(num_tokens, 4)](
        q_c,
        q_c.stride(0),
        q_norm_weight,
        q_out,
        kv_c,
        kv_c.stride(0),
        kv_norm_weight,
        kv_out,
        k_pe,
        k_pe.stride(0),
        index_k,
        index_k.stride(0),
        k_norm_weight,
        k_norm_bias,
        eps,
        mla_slot_mapping,
        mla_slot_mapping.shape[0],
        # PPU MODIFICATION: raw bytes (SW encoder), no fp8 view
        mla if _SW_FP8_CAST else mla.view(_FP8_DTYPE),
        mla.view(torch.float32),
        mla.view(torch.bfloat16),
        mla.shape[1],
        mla.stride(0),
        mla.stride(1),
        index_slot_mapping,
        index_slot_mapping.shape[0],
        index,  # PPU MODIFICATION: raw bytes (SW encoder / int8), no fp8 view
        index.view(torch.int32 if mx_scale else torch.float32),
        index.shape[1],
        index.stride(0),
        topk_indices_buffer,
        topk_indices_buffer.stride(0),
        Q_DIM=q_dim,
        Q_BLOCK=triton.next_power_of_2(q_dim),
        KV_DIM=kv_dim,
        ROPE_DIM=rope_dim,
        INDEX_DIM=index_dim,
        TOPK=topk,
        TOPK_BLOCK=min(1024, triton.next_power_of_2(topk)),
        MX_SCALE=mx_scale,
        USE_PDL=use_pdl,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        launch_pdl=use_pdl,
    )
    return q_out, kv_out


# ---------------------------------------------------------------------------
# int8 scoring redirection in vllm.model_executor.layers.sparse_attn_indexer
# ---------------------------------------------------------------------------


# Originals captured by ``_install_int8_scoring``. The replacements stay plain
# module-level functions so the patch inventory can check their signatures
# against upstream, while installation itself remains gated on the int8 path.
_original_fp8_mqa = None
_original_fp8_paged = None


def fp8_fp4_mqa_logits(
    q,
    kv,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
    max_seqlen_k=None,
    logits_dtype=torch.float32,
    **kwargs,
):
    """Score the int8 indexer Q through the SAIL int8 kernel.

    Only int8 Q without a Q scale -- the cache this patch writes -- takes the
    int8 path; any other producer (FP8 Q from a model this patch does not own)
    delegates to the captured original.
    """
    q_values, q_scale = q if isinstance(q, tuple) else (q, None)
    if q_values.dtype != torch.int8:
        if max_seqlen_k is None:
            raise ValueError(
                "fp8_fp4_mqa_logits needs max_seqlen_k; None would silently "
                "select the global-column layout."
            )
        return _original_fp8_mqa(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            max_seqlen_k,
            logits_dtype=logits_dtype,
            **kwargs,
        )
    if q_scale is not None:
        # An MXFP4 cache packs Q into uint8 (viewed as int8) and carries int32
        # block scales; scoring those bytes as int8 values would be wrong, and
        # these targets have no MXFP4 logits kernel.
        raise NotImplementedError(
            "An MXFP4 DSA indexer cache cannot be scored without FP8 MMA: the "
            "int8 path covers int8 Q with the head scale folded into the "
            "weights (q_scale=None) only."
        )
    if max_seqlen_k is None:
        raise ValueError(
            "int8 mqa logits need max_seqlen_k; None would silently select the "
            "global-column layout."
        )
    k_values, k_scales = (kv[0], kv[1]) if isinstance(kv, tuple) else (kv, None)
    # The int8 kernel emits logits in the weights' dtype: the int8 Q's
    # per-head scale is folded into the FP32 weights above, so FP32 logits
    # are what the producer contract asks for.
    return _sail_deep_gemm.int8_mqa_logits(
        q_values,
        (k_values.view(torch.int8), k_scales),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
        max_seqlen_k=max_seqlen_k,
    )


def fp8_fp4_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len=None,
    logits_dtype=torch.float32,
    indices=None,
    **kwargs,
):
    """Score the int8 indexer Q against the paged cache through the int8 kernel.

    Only int8 Q without a Q scale takes the int8 path; any other producer
    delegates to the captured original.
    """
    q_values, q_scale = q if isinstance(q, tuple) else (q, None)
    if q_values.dtype != torch.int8:
        if max_model_len is None:
            raise ValueError(
                "fp8_fp4_paged_mqa_logits needs max_model_len; None would size "
                "the logits columns from an unset bound."
            )
        return _original_fp8_paged(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            logits_dtype=logits_dtype,
            indices=indices,
            **kwargs,
        )
    if q_scale is not None:
        raise NotImplementedError(
            "An MXFP4 DSA indexer cache cannot be scored without FP8 MMA: the "
            "int8 paged path covers int8 Q with the head scale folded into the "
            "weights (q_scale=None) only."
        )
    if indices is not None:
        # Dropping the request index would silently score the varlen decode
        # schedule with the wrong rows, so refuse instead.
        raise NotImplementedError(
            "SAIL's int8 paged logits API takes no per-row request index, so a "
            "varlen decode schedule (chunk_decode) cannot be honoured here."
        )
    if max_model_len is None:
        raise ValueError(
            "int8 paged mqa logits need max_model_len; None would size the "
            "logits columns from an unset bound."
        )
    # The int8 paged API is request-major: one query row, one context and one
    # page-table row each. The producer hands over batched queries
    # ``[B, next_n, H, D]`` with per-request contexts ``[B, next_n]``
    # (next_n == 1 outside speculative decoding); the schedule metadata is
    # built over the same flattened rows, matching how the PPU op expands them
    # in ``vllm_sail/attention/ops/mla_sparse.py``.
    if q_values.dim() == 3:
        q_values = q_values.unsqueeze(1)
    batch, next_n = q_values.shape[0], q_values.shape[1]
    rows = batch * next_n
    if context_lens.numel() not in (batch, rows):
        raise ValueError(
            f"context_lens has {context_lens.numel()} entries for {batch} "
            f"requests x {next_n} queries; expected {batch} or {rows}."
        )
    if weights.shape[0] != rows:
        raise ValueError(f"weights has {weights.shape[0]} rows for {rows} queries.")
    flat_q = q_values.reshape(rows, 1, *q_values.shape[2:])
    contexts = context_lens.reshape(-1, 1).to(torch.int32)
    if contexts.shape[0] != rows:
        contexts = contexts.expand(-1, next_n).reshape(rows, 1)
    contexts = contexts.contiguous()
    tables = block_tables
    if tables.dim() >= 2 and tables.stride(-1) != 1:
        tables = tables.clone(memory_format=torch.contiguous_format)
    if tables.shape[0] not in (batch, rows):
        raise ValueError(
            f"block_tables has {tables.shape[0]} rows for {batch} requests x "
            f"{next_n} queries; expected {batch} or {rows}."
        )
    if tables.shape[0] != rows:
        tables = tables.repeat_interleave(next_n, dim=0)
    return _sail_deep_gemm.int8_paged_mqa_logits(
        flat_q,
        kv_cache,
        weights,
        contexts,
        tables,
        schedule_metadata,
        max_model_len,
        clean_logits=False,
    )


def _install_int8_scoring() -> None:
    """Rebind the sparse_attn_indexer scoring helpers for the int8 path.

    The fork branches inside ``sparse_attn_indexer`` on
    ``q.dtype == torch.int8``; instead of copying that large function, rebind
    its two ``fp8_fp4_*`` scoring helpers, which translate to the DeepGEMM
    int8 APIs, falling back to the real fp8 kernels when the caller still
    hands over FP8 tensors. ``mqa_logits_mx_inputs`` needs no replacement:
    these targets never set ``mqa_logits_use_mx_scales()`` (SM100-only), so
    upstream's transform is already the identity the int8 API wants -- and the
    attention patch layer owns that symbol here, so a second registration
    would trip the double-patch guard. That layer's logits-dtype pin skips only
    while this path is active (``dsa_indexer_gate.int8_indexer_active``).

    Both names go together: one int8 quantizer with an FP8 scorer would score
    int8 bytes as fp8. A name another patch already owns is a configuration
    error, not a crash during plugin import, so the whole install is skipped.
    """
    global _original_fp8_mqa, _original_fp8_paged

    taken = [
        name
        for name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits")
        if getattr(getattr(_sparse, name), PATCH_MARKER, None) is not None
    ]
    if taken:
        logger.warning(
            "Skipping the Kimi K3.1 int8 DSA scoring patch: %s is already "
            "patched in %s, so the int8 indexer cache would be scored by the "
            "FP8 logits kernels.",
            ", ".join(taken),
            _SPARSE_MODULE,
        )
        return

    _original_fp8_mqa = _sparse.fp8_fp4_mqa_logits
    _original_fp8_paged = _sparse.fp8_fp4_paged_mqa_logits
    patch(
        _SPARSE_MODULE,
        "fp8_fp4_mqa_logits",
        reason=(
            "Targets without FP8 MMA have no fp8 deep_gemm mqa_logits kernel; "
            "the int8 indexer's Q/K (int8 values, ue8m0 scales) score through "
            "the SAIL int8_mqa_logits wrapper with clean_logits=False, matching "
            "the fp8_fp4 contract of compressed rows with an uncleaned tail. "
            "FP8 tensors still fall back to the real fp8 kernel."
        ),
        affected_versions=_AFFECTED,
        remove_when="deep_gemm exposes a platform-dispatched mqa_logits entry point.",
    )(fp8_fp4_mqa_logits)
    patch(
        _SPARSE_MODULE,
        "fp8_fp4_paged_mqa_logits",
        reason=(
            "Targets without FP8 MMA have no fp8 deep_gemm paged mqa_logits "
            "kernel; the int8 indexer's padded Q scores through the SAIL "
            "int8_paged_mqa_logits wrapper over the same uint8 cache view "
            "(int8 values + fp32 ue8m0 scale per page), preserving the "
            "[B*next_n, max_model_len] output contract with uncleaned tails. "
            "FP8 tensors still fall back to the real fp8 kernel."
        ),
        affected_versions=_AFFECTED,
        remove_when="deep_gemm exposes a platform-dispatched paged mqa_logits entry point.",
    )(fp8_fp4_paged_mqa_logits)


if _USE_INT8_INDEXER:
    _install_int8_scoring()


# ---------------------------------------------------------------------------
# Consumer rebinding: ``nvidia/dsa.py`` imports the three entry points by
# value at module scope (``from ..ops.dsa_indexer import ...``), so patching
# the ops module alone never reaches ``KimiK3DSAIndexer``: its aliases keep the
# upstream Triton kernels, whose ``tl.float8e4nv`` stores cannot compile on
# capability (8,0), and its Q never gets quantised to int8.
# ---------------------------------------------------------------------------
_DSAMODEL_MODULE = "vllm.models.kimi_k3.nvidia.dsa"


def _rebind_kimi_consumer() -> None:
    """Rebind the model module's by-value imports to the patched entries.

    ``dsa.py`` binds the three entries with ``from ..ops.dsa_indexer import
    ...``, so patching the ops module alone never reaches
    ``KimiK3DSAIndexer``. Idempotent (a reloaded patch module re-enters here)
    and loud when an alias is already owned by something else.
    """
    consumer = importlib.import_module(_DSAMODEL_MODULE)
    unbound = []
    for name, upstream in (
        ("indexer_k_norm_quant_insert", _upstream_k_insert),
        ("indexer_q_quant_fold_weights", _upstream_q_quant),
        ("dsa_fused_prologue", _upstream_prologue),
    ):
        replacement = getattr(_dsa_indexer, name)
        current = getattr(consumer, name, None)
        if current is replacement:
            continue
        if current is not upstream:
            unbound.append(name)
            continue
        setattr(consumer, name, replacement)
    if unbound:
        logger.warning(
            "%s still binds %s to another implementation, so the 810E DSA "
            "indexer overrides do not reach this model module.",
            _DSAMODEL_MODULE,
            ", ".join(unbound),
        )


if _SW_FP8_CAST or _USE_INT8_INDEXER:
    _rebind_kimi_consumer()
