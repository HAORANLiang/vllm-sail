# SPDX-License-Identifier: Apache-2.0
"""Gather PPU's packed-uint16 MXFP4 scales as raw bytes.

PPU's MXFP4 MoE kernels pack two UE8M0 block scales into one ``torch.uint16``
element, and expert-parallel dispatch moves those scale tensors across ranks
with all-gather. ``ncclDataTypeEnum.from_torch`` has no uint16 entry, so the
gather raises ``ValueError: Unsupported dtype torch.uint16`` before any byte
moves.

``ncclAllGather`` / ``ncclBroadcast`` only consume ``data_ptr()`` + count +
dtype label and never inspect strides, so re-labelling the buffer as uint8
and doubling the element count is an exact byte move -- including the
col-major (transposed) scale layouts whose ``stride(-1) != 1`` makes
``.view(torch.uint8)`` itself raise. The tensor is never touched.

Only the two pure-byte gather entries are replaced. Arithmetic collectives
(``all_reduce``, ``reduce_scatter``) stay upstream: reducing packed scales
byte-wise would mix the two scales inside each uint16 and corrupt them.

Leaf-style module: importable on a bare CPU runner, patches install when
``install()`` runs. Bodies are copied verbatim from upstream except for the
count/dtype computation and bound to the upstream module's globals, so live
helper lookups keep working.
"""

from __future__ import annotations

import importlib

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_COMMUNICATOR = "vllm.distributed.device_communicators.pynccl"
_AFFECTED = ">=0.34.0,<0.35.0"

REASON_GATHER = (
    "PPU packs two MXFP4 UE8M0 scales per torch.uint16 element and "
    "expert-parallel dispatch all-gathers those scales, but NCCL has no "
    "uint16 dtype so from_torch raises. ncclAllGather moves raw bytes from "
    "data_ptr() and never inspects strides, so re-labelling as uint8 with "
    "double the count is exact even for transposed scale layouts."
)
REASON_GATHERV = (
    "Same packed-uint16 MXFP4 scale gather as all_gather, through the "
    "per-split ncclBroadcast loop: relabel each split as uint8 with double "
    "the element count; the buffers themselves are never touched."
)
REMOVE_WHEN = (
    "upstream pynccl (or the NCCL dtype table) carries packed uint16 scale "
    "buffers through all-gather without a dtype error, or PPU stops packing "
    "MXFP4 scales into uint16."
)

METADATA = (
    (
        f"{_COMMUNICATOR}.PyNcclCommunicator.all_gather",
        REASON_GATHER,
        _AFFECTED,
        REMOVE_WHEN,
    ),
    (
        f"{_COMMUNICATOR}.PyNcclCommunicator.all_gatherv",
        REASON_GATHERV,
        _AFFECTED,
        REMOVE_WHEN,
    ),
)

_installed = False
#: True when the upstream module cannot be imported; the fix has no consumer.
_unavailable = False


def install() -> None:
    """Register the uint16-safe gather replacements once."""
    global _installed, _unavailable
    if _installed or _unavailable:
        return

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    try:
        pynccl = importlib.import_module(_COMMUNICATOR)
    except ImportError as exc:
        _unavailable = True
        logger.warning(
            "Skipping the packed-uint16 gather patch: the upstream pynccl "
            "module cannot be imported (%s).",
            exc,
        )
        return

    def _with_target_globals(fn):
        return bind_body(fn, pynccl)

    @patch(
        _COMMUNICATOR,
        "PyNcclCommunicator.all_gather",
        reason=REASON_GATHER,
        affected_versions=_AFFECTED,
        remove_when=REMOVE_WHEN,
    )
    @_with_target_globals
    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        if input_tensor.dtype == torch.uint16:
            # PPU packs two MXFP4 UE8M0 scales per uint16; NCCL has no uint16
            # dtype. The call moves raw bytes and never inspects strides, so
            # uint8 with double the count is exact for any layout.
            count = input_tensor.numel() * 2
            dtype = ncclDataTypeEnum.from_torch(torch.uint8)
        else:
            count = input_tensor.numel()
            dtype = ncclDataTypeEnum.from_torch(input_tensor.dtype)
        self.nccl.ncclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            count,
            dtype,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    @patch(
        _COMMUNICATOR,
        "PyNcclCommunicator.all_gatherv",
        reason=REASON_GATHERV,
        affected_versions=_AFFECTED,
        remove_when=REMOVE_WHEN,
    )
    @_with_target_globals
    def all_gatherv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        assert output_tensor.shape[0] == sum(sizes)
        if input_tensor.dtype == torch.uint16:
            # Same packed-scale relabel as all_gather, per broadcast split.
            element_size = 2
            dtype = ncclDataTypeEnum.from_torch(torch.uint8)
        else:
            element_size = 1
            dtype = ncclDataTypeEnum.from_torch(input_tensor.dtype)
        split_offset = 0
        self.nccl.ncclGroupStart()
        for root, split_size in enumerate(sizes):
            dst_slice = output_tensor[split_offset : split_offset + split_size]
            self.nccl.ncclBroadcast(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(dst_slice.data_ptr()),
                dst_slice.numel() * element_size,
                dtype,
                root,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            split_offset += split_size
        self.nccl.ncclGroupEnd()

    _installed = True