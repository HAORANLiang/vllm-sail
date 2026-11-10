# SPDX-License-Identifier: Apache-2.0
"""NVTX instrumentation for ``vllm/v1/core/sched/scheduler.py``.

Port of the fork's scheduler hunks (+34 lines) as delegating patches, per
``docs/developer_guide/architecture.md``:

* ``@annotate("schedule")`` on ``Scheduler.schedule`` — exact port.
* ``sche_mark(scheduler_output)`` — the fork called it inside
  ``_update_after_schedule`` *and* redundantly from engine core / model
  runner. ``schedule()`` calls ``_update_after_schedule`` on every path, so
  marking here covers all consumers with a single call site.

Unlike the fork, this preserves the 6-line upstream comment above the
routed-experts block-ID snapshot that the fork's NVTX commit accidentally
deleted (noted in §1.6), because no upstream body is copied.

Installed only by ``vllm_sail.profiling.install()`` when
``VLLM_SAIL_NVTX_PROFILE`` is set; never imported otherwise.
"""

from __future__ import annotations

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_sail.patch.utils import patch
from vllm_sail.profiling import nvtx

_AFFECTED = ">=0.33.0,<0.34.0"

_upstream_schedule = Scheduler.schedule
_upstream_update_after_schedule = Scheduler._update_after_schedule


@patch(
    "vllm.v1.core.sched.scheduler",
    "Scheduler.schedule",
    reason=(
        "[profiling, opt-in] NVTX `schedule` range from the in-tree PPU fork "
        "(fork scheduler.py hunk @@ -436). Installed only when "
        "VLLM_SAIL_NVTX_PROFILE is set; the wrapper delegates to upstream."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) "
        "is retired, or upstream adds native NVTX/tracing hooks to the v1 "
        "scheduler."
    ),
)
@nvtx.annotate("schedule")
def schedule(self, *args, **kwargs):
    return _upstream_schedule(self, *args, **kwargs)


@patch(
    "vllm.v1.core.sched.scheduler",
    "Scheduler._update_after_schedule",
    reason=(
        "[profiling, opt-in] Emits the fork's sche_mark NVTX events (new/ "
        "finished requests, scheduled token counts) for every scheduler "
        "output. Centralised here because schedule() calls this method on "
        "every path, replacing the fork's three duplicated call sites."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the opt-in NVTX subpackage (docs/developer_guide/architecture.md) "
        "is retired, or upstream adds native NVTX/tracing hooks to the v1 "
        "scheduler."
    ),
)
def _update_after_schedule(self, scheduler_output):
    _upstream_update_after_schedule(self, scheduler_output)
    nvtx.sche_mark(scheduler_output)
