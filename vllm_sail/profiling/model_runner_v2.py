# SPDX-License-Identifier: Apache-2.0
"""NVTX instrumentation for the model runner.

``vllm.v1.worker.gpu.model_runner.GPUModelRunner`` is instrumented at
whole-method boundaries without copying any upstream body:

* ``[prof_range]: iter N`` + ``prof_iter`` + ``sche_mark`` -> wrapper around
  ``execute_model``.
* ``total bs=..., P bs=...`` -> the same wrapper. The runner inlines the forward
  into ``execute_model`` (there is no ``_model_forward`` to wrap), so this range
  coincides with ``[prof_range]`` instead of nesting strictly inside it.
* ``[FW_NATIVE] op:sampler,...`` -> wrapper around ``sample``. The runner
  computes the logits inside ``sample`` and never exposes them, so the label
  reports the sampling input's shape; the logits shape is still recorded by the
  ``[FW_GEMM] op:compute_logits`` range.
* ``@annotate("prepare_inputs")`` on ``prepare_inputs``.
* ``compute_logits`` ranges are installed lazily from the ``execute_model``
  wrapper.

The step splits in two: ``execute_model`` runs the forward and ``sample_tokens``
runs the sampling. Only the forward gets the iteration range; the sampling work
stays visible through the sampler and compute-logits ranges above, and the
per-step ``[prof_range]`` emitted by the engine core still spans both.

Dummy runs (memory profiling and CUDA-graph capture) call ``execute_model``
with ``dummy_run=True``, which bypasses every instrumented method, so they are
skipped here too: warmup must neither emit ranges nor consume iteration numbers.
"""

from __future__ import annotations

from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_sail.patch.utils import patch
from vllm_sail.profiling import nvtx

_AFFECTED = ">=0.32.0,<0.33.0"

_UPSTREAM_MODULE = "vllm.v1.worker.gpu.model_runner"

_REMOVE_WHEN = (
    "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) is "
    "retired, or upstream adds native NVTX/tracing hooks to the V2 model "
    "runner."
)

_upstream_prepare_inputs = GPUModelRunner.prepare_inputs
_upstream_sample = GPUModelRunner.sample
_upstream_execute_model = GPUModelRunner.execute_model


@patch(
    _UPSTREAM_MODULE,
    "GPUModelRunner.prepare_inputs",
    reason=(
        "[profiling, opt-in] NVTX `prepare_inputs` range. The wrapper delegates "
        "to upstream unchanged."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
@nvtx.annotate("prepare_inputs")
def prepare_inputs(self, *args, **kwargs):
    return _upstream_prepare_inputs(self, *args, **kwargs)


@patch(
    _UPSTREAM_MODULE,
    "GPUModelRunner.sample",
    reason=(
        "[profiling, opt-in] Wraps sampling in the fork's "
        "`[FW_NATIVE] op:sampler,...` range. The runner does not expose the "
        "logits tensor, so the label carries the hidden-state shape instead."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def sample(self, hidden_states, *args, **kwargs):
    shape = getattr(hidden_states, "shape", None)
    nvtx.range_push(f"[FW_NATIVE] op:sampler,hidden_states:{shape}")
    try:
        return _upstream_sample(self, hidden_states, *args, **kwargs)
    finally:
        nvtx.range_pop()


@patch(
    _UPSTREAM_MODULE,
    "GPUModelRunner.execute_model",
    reason=(
        "[profiling, opt-in] Wraps one worker iteration in the fork's "
        "`[prof_range]: iter N` range, the `total bs=..., P bs=...` range, the "
        "prof_iter hook and sche_mark. Without it the worker emits no NVTX at "
        "all. Dummy runs are forwarded untouched."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def execute_model(
    self,
    scheduler_output,
    intermediate_tensors=None,
    dummy_run=False,
    *args,
    **kwargs,
):
    if dummy_run:
        return _upstream_execute_model(
            self, scheduler_output, intermediate_tensors, dummy_run, *args, **kwargs
        )

    nvtx.instrument_compute_logits(getattr(self, "model", None))
    nvtx.sche_mark(scheduler_output)
    # Lazy initialization: the runner may have been constructed while the
    # profiling patches were still being installed (see the engine-core
    # counterpart), so the attribute is created on first use.
    iteration = getattr(self, "iteration", 0)
    nvtx.prof_iter(iteration)
    nvtx.range_push(f"[prof_range]: iter {iteration}")
    nvtx.range_push(nvtx.batch_label(scheduler_output))
    self.iteration = iteration + 1
    try:
        return _upstream_execute_model(
            self, scheduler_output, intermediate_tensors, dummy_run, *args, **kwargs
        )
    finally:
        nvtx.range_pop()
        nvtx.range_pop()
