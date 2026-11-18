# SPDX-License-Identifier: Apache-2.0
"""Drop rescale/normalize flags the Kimi-K2.5 checkpoint processor cannot take.

vLLM's multimodal budget profiling calls the processor with
``do_rescale=False, do_normalize=False``. The wrapper forwards both flags to
the checkpoint's remote vision processor, whose ``preprocess`` only accepts
``(medias, return_tensors)`` and does its own fixed rescale/normalize, so the
call dies with a TypeError on transformers 5.16.1. The wrapper declares the
flags as defaulted parameters and always forwards them, so dropping them from
the outer call is not enough: shim ``image_processor.preprocess`` for the
duration of the call and strip the flags it rejects.
"""

from __future__ import annotations

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.34.0,<0.35.0"
_REASON = (
    "The Kimi-K2.5 checkpoint vision processor's preprocess takes no "
    "do_rescale/do_normalize; forwarding vLLM's profiling flags raises "
    "TypeError during multimodal budget computation."
)
_REMOVE_WHEN = (
    "The checkpoint processor accepts the flags, or vLLM stops passing them "
    "to processors that do not declare them."
)
_MODULE = "vllm.transformers_utils.processors.kimi_k25"

from vllm.transformers_utils.processors import kimi_k25 as _kimi_k25  # noqa: E402

_original_call = _kimi_k25.KimiK25Processor.__call__
_FLAG_NAMES = ("do_rescale", "do_normalize")


@patch(
    _MODULE,
    "KimiK25Processor.__call__",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def __call__(self, *args, **kwargs):
    ip = self.image_processor
    orig_preprocess = ip.preprocess

    def preprocess(*a, **kw):
        while True:
            try:
                return orig_preprocess(*a, **kw)
            except TypeError as exc:
                msg = str(exc)
                dropped = [n for n in _FLAG_NAMES if n in msg and n in kw]
                if not dropped:
                    raise
                for name in dropped:
                    kw.pop(name, None)

    had_own = "preprocess" in vars(ip)
    own = vars(ip).get("preprocess")
    ip.preprocess = preprocess
    try:
        return _original_call(self, *args, **kwargs)
    finally:
        if had_own:
            ip.preprocess = own
        else:
            del ip.preprocess