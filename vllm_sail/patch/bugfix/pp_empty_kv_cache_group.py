# SPDX-License-Identifier: Apache-2.0
"""Drop KV cache groups that project to zero layers on a PP worker.

Qwen4Exp's PLE n-gram embedding layers carry a ``MambaSpec`` with prefill
checkpoint parameters, which compares unequal to the plain hybrid-layer spec,
so they form their own KV cache group -- and that group lives only on the
first pipeline-parallel stage. Projecting the global groups onto any other
stage yields an empty ``layer_names`` list, but upstream still appends the
group while keeping the **unfiltered global** ``UniformTypeKVCacheSpecs``.
``get_kv_cache_config_from_groups`` then materializes tensors from
``group_spec.kv_cache_specs.items()`` for layers this stage does not own, and
``allocate_kv_cache`` dies with ``StopIteration`` looking the group up.

The fix drops a group entirely when none of its layers belong to this worker:
an empty group contributes nothing, and the per-worker consistency assert in
``get_kv_cache_configs`` sums layer counts, so skipping it stays balanced.
Groups with at least one local layer keep upstream's behavior verbatim.

Leaf-style module: importable on a bare CPU runner, the patch installs when
``install()`` runs. The replacement wraps the upstream projection and filters
its result, so the non-empty branch never needs a copied body.
"""

from __future__ import annotations

import importlib

from vllm_sail.patch.utils import patch

_PLANNING_MODULE = "vllm.v1.core.kv_cache_planning"
_TARGET = "_project_kv_cache_groups_to_worker"
_AFFECTED = ">=0.34.0,<0.35.0"

REASON = (
    "Under pipeline parallelism a KV cache group owned by another stage "
    "projects to zero local layers (e.g. the Qwen4Exp PLE n-gram group, which "
    "exists only on the first stage). Upstream keeps the empty group with the "
    "unfiltered global UniformTypeKVCacheSpecs, so the config builder "
    "materializes tensors for layers this worker does not own and "
    "allocate_kv_cache raises StopIteration. Drop groups with no local "
    "layers; they contribute nothing to this worker."
)
REMOVE_WHEN = (
    "upstream _project_kv_cache_groups_to_worker skips groups whose "
    "projection onto the worker is empty (verify by reading the function)."
)

METADATA = ((f"{_PLANNING_MODULE}.{_TARGET}", REASON, _AFFECTED, REMOVE_WHEN),)

_installed = False
#: True when the upstream module cannot be imported; the fix has no consumer.
_unavailable = False


def install() -> None:
    """Register the empty-group filter once."""
    global _installed, _unavailable
    if _installed or _unavailable:
        return

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    try:
        planning = importlib.import_module(_PLANNING_MODULE)
    except ImportError as exc:
        _unavailable = True
        logger.warning(
            "Skipping the empty PP KV cache group patch: the upstream "
            "kv_cache_planning module cannot be imported (%s).",
            exc,
        )
        return

    _upstream_project = planning._project_kv_cache_groups_to_worker

    @patch(
        _PLANNING_MODULE,
        _TARGET,
        reason=REASON,
        affected_versions=_AFFECTED,
        remove_when=REMOVE_WHEN,
    )
    def _project_kv_cache_groups_to_worker(
        global_kv_cache_groups, worker_spec
    ):
        projected = _upstream_project(global_kv_cache_groups, worker_spec)
        return [group for group in projected if group.layer_names]

    _installed = True