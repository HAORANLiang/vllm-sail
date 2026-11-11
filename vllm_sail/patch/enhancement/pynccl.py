# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
"""Keep basic PyNccl communication usable without PCCL suspend/resume APIs.

PPU's NCCL-compatible library may omit communicator memory suspension. Upstream
already tolerates those missing symbols on ROCm; PPU needs the same loader
policy. All other symbol requirements and the per-library caches stay upstream.
"""

from __future__ import annotations

from vllm.distributed.device_communicators import pynccl as _pynccl
from vllm.distributed.device_communicators import pynccl_wrapper as _wrapper
from vllm.platforms import current_platform

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.31.0,<0.32.0"
_WRAPPER = "vllm.distributed.device_communicators.pynccl_wrapper"
_COMMUNICATOR = "vllm.distributed.device_communicators.pynccl"
_upstream_suspend = _pynccl.PyNcclCommunicator.suspend


def _with_target_globals(fn):
    return bind_body(fn, _wrapper)


@patch(
    _WRAPPER,
    "NCCLLibrary.__init__",
    reason=(
        "SAIL's NCCL-compatible PCCL library may omit ncclCommSuspend and "
        "ncclCommResume. Upstream tolerates this only on ROCm, so loading "
        "the PPU library raises and silently disables PyNccl."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "Upstream permits missing ncclCommSuspend/ncclCommResume on PPU, "
        "or the minimum supported SAIL SDK exports both symbols."
    ),
)
@_with_target_globals
def __init__(self, so_file: str | None = None):
    so_file = so_file or find_nccl_library()

    try:
        if so_file not in NCCLLibrary.path_to_dict_mapping:
            lib = ctypes.CDLL(so_file)
            NCCLLibrary.path_to_library_cache[so_file] = lib
        self.lib = NCCLLibrary.path_to_library_cache[so_file]
    except Exception as e:
        logger.error(
            "Failed to load NCCL library from %s. "
            "It is expected if you are not running on NVIDIA/AMD GPUs."
            "Otherwise, the nccl library might not exist, be corrupted "
            "or it does not support the current platform %s. "
            "If you already have the library, please set the "
            "environment variable VLLM_NCCL_SO_PATH"
            " to point to the correct nccl library path.",
            so_file,
            platform.platform(),
        )
        raise e

    if so_file not in NCCLLibrary.path_to_dict_mapping:
        _funcs: dict[str, Any] = {}
        for func in NCCLLibrary.exported_functions:
            try:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            except AttributeError:
                if func.name in [
                    "ncclCommWindowRegister",
                    "ncclCommWindowDeregister",
                ]:
                    if envs.VLLM_USE_NCCL_SYMM_MEM:
                        logger.warning_once(
                            "The symbol %s is not found in the NCCL "
                            "library %s. To enable VLLM_USE_NCCL_SYMM_MEM "
                            " please update your NCCL version to >= "
                            "2.27.03.",
                            func.name,
                            so_file,
                        )
                    if current_platform.is_rocm():
                        # Having an exception here on ROCm platform is
                        # not allowed during graph capturing
                        continue
                elif func.name == "ncclCommQueryProperties":
                    # Optional on NCCL versions older than 2.29.
                    continue
                elif func.name in ("ncclCommSuspend", "ncclCommResume"):
                    # RCCL doesn't export these; NCCL >= 2.29.7 does, and
                    # vLLM's CUDA path already requires that version.
                    # PyNcclCommunicator checks has_symbol() before
                    # calling either.
                    # PPU MODIFICATION: begin
                    if current_platform.is_rocm() or current_platform.is_ppu():
                        continue
                    # PPU MODIFICATION: end
                raise
        NCCLLibrary.path_to_dict_mapping[so_file] = _funcs
    self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]


@patch(
    _COMMUNICATOR,
    "PyNcclCommunicator.suspend",
    reason=(
        "PPU must have both ncclCommSuspend and ncclCommResume before "
        "suspending a communicator; upstream checks only the former."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "Upstream checks both suspend/resume symbols before suspension, "
        "or the minimum supported SAIL SDK exports both symbols."
    ),
)
def suspend(self):
    if current_platform.is_ppu() and not self.disabled and not self._suspended:
        if not all(
            self.nccl.has_symbol(name)
            for name in ("ncclCommSuspend", "ncclCommResume")
        ):
            _pynccl.logger.warning_once(
                "PCCL does not export both ncclCommSuspend and ncclCommResume; "
                "skipping communicator memory suspension on PPU."
            )
            return
    return _upstream_suspend(self)


_upstream_all_gather = _pynccl.PyNcclCommunicator.all_gather
_upstream_all_gatherv = _pynccl.PyNcclCommunicator.all_gatherv


def _gather_transport_views(output_tensor, input_tensor):
    import torch

    if current_platform.is_ppu() and input_tensor.dtype == torch.uint16:
        # Gather only copies bytes. An equal-width BF16 view preserves packed
        # E8M0 scale bits, tensor shapes, counts and the caller's output dtype.
        return output_tensor.view(torch.bfloat16), input_tensor.view(torch.bfloat16)
    return output_tensor, input_tensor


_GATHER_META = dict(
    reason="PCCL gathers need a supported transport dtype for packed uint16 MXFP4 scales.",
    affected_versions=_AFFECTED,
    remove_when="PyNccl and PCCL support uint16 in all_gather and all_gatherv.",
)


@patch(_COMMUNICATOR, "PyNcclCommunicator.all_gather", **_GATHER_META)
def all_gather(self, output_tensor, input_tensor, stream=None):
    output_tensor, input_tensor = _gather_transport_views(output_tensor, input_tensor)
    return _upstream_all_gather(self, output_tensor, input_tensor, stream=stream)


@patch(_COMMUNICATOR, "PyNcclCommunicator.all_gatherv", **_GATHER_META)
def all_gatherv(self, output_tensor, input_tensor, sizes, stream=None):
    output_tensor, input_tensor = _gather_transport_views(output_tensor, input_tensor)
    return _upstream_all_gatherv(self, output_tensor, input_tensor, sizes, stream=stream)
