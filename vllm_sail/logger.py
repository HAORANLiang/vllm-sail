# SPDX-License-Identifier: Apache-2.0
"""Logging glue for ``vllm_sail`` modules.

vLLM configures only its own ``vllm`` logger namespace (formatter, level,
``propagate=False``) and only when an entrypoint or executor process calls
``configure_logging*`` -- which can be long after this plugin is imported.
Loggers outside that namespace, i.e. every ``vllm_sail.*`` module logger
created with ``init_logger(__name__)``, bubble to an unconfigured root where
INFO records are silently dropped (WARNING and up survive through the root's
last-resort stderr handler).

``configure_sail_logging`` attaches a delegating handler to the
``vllm_sail`` namespace instead of snapshotting vLLM's handlers: a snapshot
would be replaced -- and left stale -- by vLLM's later ``dictConfig``. The
delegate forwards each record to whatever handlers the ``vllm`` logger has
*at emit time*, gated by their levels, so ``init_logger(__name__)`` in any
vllm_sail module behaves exactly like vLLM's own loggers: INFO and up follow
``VLLM_LOGGING_LEVEL`` and vLLM's formatter once logging is configured, and
are dropped like vLLM's own records before that.
"""

import logging

_SAIL_NAMESPACE = "vllm_sail"
_VLLM_NAMESPACE = "vllm"


class _VllmDelegateHandler(logging.Handler):
    """Forward records to vLLM's currently-configured handlers.

    The gate is the *handler* level, which is where vLLM's dictConfig puts
    ``VLLM_LOGGING_LEVEL``; the ``vllm`` logger itself is checked only for
    being configured at all. ``Handler.handle`` runs the handler's filters
    and formatter, so the record is emitted exactly as vLLM's own would be.
    """

    def emit(self, record: logging.LogRecord) -> None:
        vllm_logger = logging.getLogger(_VLLM_NAMESPACE)
        for handler in vllm_logger.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)


def configure_sail_logging() -> None:
    """Route ``vllm_sail.*`` records through vLLM's logging configuration.

    Idempotent and safe to call in every process; called at
    ``vllm_sail`` package import. The namespace logger's own level is set to
    DEBUG so every record reaches the delegate -- the visibility gate is the
    vLLM handler level applied at emit time. ``propagate`` is disabled so a
    record is never emitted twice (vLLM's namespace logger also propagates
    nowhere).
    """
    sail_logger = logging.getLogger(_SAIL_NAMESPACE)
    if not any(
        isinstance(handler, _VllmDelegateHandler)
        for handler in sail_logger.handlers
    ):
        sail_logger.addHandler(_VllmDelegateHandler())
        sail_logger.setLevel(logging.DEBUG)
    sail_logger.propagate = False
