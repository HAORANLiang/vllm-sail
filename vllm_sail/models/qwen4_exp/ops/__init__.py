# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp QSA indexer kernels for PPU.

This ``__init__`` deliberately imports nothing, matching
``vllm_sail/models/deepseek_v4/ops/__init__.py``: the leaf modules import
``torch``, ``vllm.triton_utils`` and DeepGEMM at module top level. Import
them directly, under the QSA selection gate.
"""
