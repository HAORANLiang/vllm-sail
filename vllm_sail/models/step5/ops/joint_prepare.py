# SPDX-License-Identifier: Apache-2.0
"""Opt-in launch tuning for the unchanged Step5 joint prepare kernel."""

from __future__ import annotations

import torch
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.models.step5.ops.soft_fp8 import ByteFP8Launch


class JointPrepareLaunch(ByteFP8Launch):
    """Use eight warps only for the validated TP4 MTP preparation geometry."""

    @staticmethod
    def _eligible(grid, args, kwargs):
        if not envs.VLLM_SAIL_STEP5_JOINT_PREPARE or not current_platform.is_ppu():
            return False
        if len(args) != 60 or not isinstance(grid, tuple) or len(grid) != 2:
            return False
        tokens = grid[0]
        if type(tokens) is not int or tokens not in (32, 64) or grid[1] != 12:
            return False
        if kwargs != {"num_warps": 4, "enable_fp_fusion": False}:
            return False
        if args[16] is not False or any(value is not None for value in args[11:16]):
            return False
        constants = {
            17: 0,
            18: 0,
            19: 0,
            20: 8,
            21: 0,
            22: 8,
            32: 1e-5,
            50: tokens,
            51: 64,
            55: 1e-5,
        }
        if any(args[index] != value for index, value in constants.items()):
            return False
        if args[26:30] != (4, 8, 1, tokens) or args[30] != 32:
            return False
        if args[33:35] != (1.0, False) or args[45:47] != (16, 1):
            return False
        if args[48] != 64 or args[56] != 1.0 or args[59] is not True:
            return False
        expected = {
            0: (torch.bfloat16, (tokens, 1024)),
            1: (torch.bfloat16, (tokens, 256)),
            2: (torch.bfloat16, (tokens, 4)),
            3: (torch.bfloat16, (256,)),
            4: (torch.bfloat16, (256,)),
            5: (torch.bfloat16, (256,)),
            7: (torch.int64, (tokens,)),
            8: (torch.float8_e4m3fn, (1, tokens, 8, 256)),
            9: (torch.bfloat16, (tokens, 1, 256)),
            10: (torch.float32, (1, tokens, 8)),
            35: (torch.bfloat16, (tokens, 3456)),
            36: (torch.float32, (192,)),
            37: (torch.float32, (192,)),
            39: (torch.int64, (tokens,)),
            40: (torch.bfloat16, (tokens, 3072)),
            41: (torch.bfloat16, (tokens, 192)),
            42: (torch.bfloat16, (tokens, 192)),
            57: (torch.float32, ()),
            58: (torch.float32, ()),
        }
        query = args[0]
        if not isinstance(query, torch.Tensor) or not query.is_cuda:
            return False
        for index, (dtype, shape) in expected.items():
            tensor = args[index]
            if not isinstance(tensor, torch.Tensor):
                return False
            if tensor.device != query.device or tensor.dtype != dtype:
                return False
            if tuple(tensor.shape) != shape or (shape and tensor.stride(-1) != 1):
                return False
        for index, width, stride_index in ((6, 32, 31), (38, 64, 49)):
            rope = args[index]
            if not isinstance(rope, torch.Tensor) or rope.device != query.device:
                return False
            if rope.dtype != torch.bfloat16 or rope.ndim != 2:
                return False
            if rope.shape[1] != width or rope.stride(1) != 1:
                return False
            if args[stride_index] != rope.stride(0):
                return False
        if any(not args[index].is_contiguous() for index in (8, 9, 10, 40, 41, 42)):
            return False
        if args[23] != query.stride(0) or args[24] != args[1].stride(0):
            return False
        if args[25] != args[2].stride(0) or args[47] != args[35].stride(0):
            return False
        for index in (43, 44):
            cache = args[index]
            if not isinstance(cache, torch.Tensor) or cache.device != query.device:
                return False
            if cache.dtype != torch.float8_e4m3fn or cache.ndim != 4:
                return False
            if tuple(cache.shape[1:]) != (64, 1, 192) or cache.stride(-1) != 1:
                return False
            if tuple(cache.stride(i) for i in range(3)) != args[52:55]:
                return False
        return tuple(torch.cuda.get_device_capability(query.device)) == (8, 9)

    def __getitem__(self, grid):
        baseline = super().__getitem__(grid)

        def run(*args, **kwargs):
            if self._eligible(grid, args, kwargs):
                kwargs = dict(kwargs, num_warps=8)
            return baseline(*args, **kwargs)

        return run
