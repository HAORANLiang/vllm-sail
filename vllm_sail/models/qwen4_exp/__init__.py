# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU kernels and selection glue for the Qwen4Exp QSA owner.

Importing this package pulls in nothing: the leaf modules import ``torch``,
``triton``, DeepGEMM and ``vllm`` at module top level and are reached only
from the runtime patches in ``vllm_sail/patch/enhancement/models/``.
"""
