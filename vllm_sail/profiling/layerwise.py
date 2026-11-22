# SPDX-License-Identifier: Apache-2.0
"""为 v0.34 Model Runner V2 的 target 和 draft/MTP 注册 Layer NVTX。

复用 --enable-layerwise-nvtx-tracing 和上游 marker helper，不依赖
VLLM_SAIL_NVTX_PROFILE 或 PyPI nvtx。首次 target/draft tracing 完成后
注册 hooks，不改变模型、采样、编译或 CUDA Graph 配置。
"""

from __future__ import annotations

import functools
import importlib

from vllm.logger import init_logger

from vllm_sail.patch.utils import patch

logger = init_logger(__name__)

_AFFECTED = ">=0.34.0,<0.35.0"
_REMOVE_WHEN = "Upstream V2 natively supports draft/MTP Layer NVTX and deduplicates shared modules."
_INSTALLED: set[str] = set()


def _make_hooks():
    """Reuse v1 input/output labels; only dedupe registration and close on errors."""
    import torch
    from vllm.utils.nvtx_pytorch_hooks import construct_marker_dict_and_push

    class SharedModuleHooks:
        def __init__(self):
            self.module_to_name_map = {}

        def module_fwd_pre_hook(self, module, args, kwargs):
            construct_marker_dict_and_push(
                f"{self.module_to_name_map[module]}(input)", module, args, kwargs
            )

        def module_fwd_hook(self, module, args, output):
            torch.cuda.nvtx.range_pop()
            construct_marker_dict_and_push(
                f"{self.module_to_name_map[module]}(output)",
                module,
                in_tensor=None,
                out_tensor=output,
            )
            torch.cuda.nvtx.range_pop()

        def register_hooks(self, network_model, module_prefix="top"):
            skip_types = (
                torch.nn.Identity,
                torch.nn.Dropout,
                torch.nn.Dropout1d,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
            )
            for name, module in network_model.named_modules(prefix=module_prefix):
                if isinstance(module, skip_types) or module in self.module_to_name_map:
                    continue
                # Register target first, then walk draft; shared layers keep the
                # first name.
                self.module_to_name_map[module] = name
                module.register_forward_pre_hook(
                    self.module_fwd_pre_hook, with_kwargs=True
                )
                module.register_forward_hook(self.module_fwd_hook, always_call=True)

    return SharedModuleHooks()


def register_layerwise_nvtx_hooks(self) -> None:
    """Register where the runner finished its first tracing; no model walk when off."""
    config = self.vllm_config
    if not config.observability_config.enable_layerwise_nvtx_tracing:
        return
    if getattr(self, "layerwise_nvtx_hooks_registered", False):
        return

    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    compilation = self.compilation_config
    if compilation.mode == CompilationMode.STOCK_TORCH_COMPILE:
        logger.warning_once(
            "STOCK_TORCH_COMPILE does not support per-layer NVTX hooks; skipping registration. "
            "Use --enforce-eager to see per-layer markers."
        )
        return
    if (
        compilation.mode != CompilationMode.NONE
        or compilation.cudagraph_mode != CUDAGraphMode.NONE
    ):
        logger.warning_once(
            "Compiled paths may bypass module hooks, and CUDA Graph replay does not rerun "
            "per-layer NVTX hooks. Use --enforce-eager to see per-layer markers."
        )

    # 注册前解析 draft 类型，避免可选导入失败后只注册部分 hooks。
    # 非最后一个 PP rank 只注册 target。
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    proposer = getattr(self, "speculator", None)
    draft = proposer.model if isinstance(proposer, DraftModelSpeculator) else None
    hooks = _make_hooks()
    hooks.register_hooks(self.model, self.model.__class__.__name__)
    if draft is not None:
        hooks.register_hooks(draft, f"draft.{draft.__class__.__name__}")
    self._sail_layerwise_nvtx_hooks = hooks
    self.layerwise_nvtx_hooks_registered = True


def install() -> None:
    """Install the config-gated compat entries; idempotent, independent of the
    outer profiling env var.
    """
    for module_path in ("vllm.v1.worker.gpu.model_runner",):
        if module_path in _INSTALLED:
            continue
        try:
            module = importlib.import_module(module_path)
            runner = module.GPUModelRunner
            # 安装前确认 dummy-run 入口；其返回时 target/draft tracing 已完成。
            original_dummy_run = runner._dummy_run

            @functools.wraps(register_layerwise_nvtx_hooks)
            def register(self):
                return register_layerwise_nvtx_hooks(self)

            patch(
                module_path,
                "GPUModelRunner._register_layerwise_nvtx_hooks",
                reason="[profiling, config-gated] Per-layer markers for V2 target and draft/MTP with shared-module dedup.",
                affected_versions=_AFFECTED,
                remove_when=_REMOVE_WHEN,
                allow_missing=True,
            )(register)
            _install_v2_dummy_run(module_path, original_dummy_run)
            _INSTALLED.add(module_path)
        except Exception as exc:
            # A missing optional dependency or upstream API drift must not block
            # engine startup.
            logger.warning(
                "Layer NVTX compat entry not installed on %s: %s", module_path, exc
            )


def _install_v2_dummy_run(module_path, original):
    @patch(
        module_path,
        "GPUModelRunner._dummy_run",
        reason="[profiling, config-gated] Register Layer NVTX after the first V2 target/draft dummy forwards finish.",
        affected_versions=_AFFECTED,
        remove_when=_REMOVE_WHEN,
    )
    @functools.wraps(original)
    def dummy_run(self, *args, **kwargs):
        # Do not register when execute_model(dummy_run=True) returns: draft has not
        # traced yet.
        result = original(self, *args, **kwargs)
        self._register_layerwise_nvtx_hooks()
        return result
