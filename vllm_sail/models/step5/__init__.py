# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-side pieces of Step5.

Unlike ``vllm_sail/models/deepseek_v4``, Step5 registers no model class through
``ModelRegistry.register_model``: vLLM ships the full ``vllm/models/step5`` tree
and the plugin only replaces individual kernels. The entry points live in
``vllm_sail/patch/``.

This ``__init__`` deliberately imports nothing: the submodules import torch and
triton at module level, and the ``vllm_sail.patch`` install order requires
importing this package to have no side effects.
"""
