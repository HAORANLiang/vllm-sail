# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-optimized Step5 DSA kernels.

These modules import torch, triton and upstream ``vllm.models.step5`` symbols at
module level. Callers reach them through
``vllm_sail/patch/performance/step5_dsa.py``, which imports them lazily on first
use.

This ``__init__`` deliberately imports nothing, so importing ``vllm_sail.models``
never pulls in the triton compilation stack.
"""
