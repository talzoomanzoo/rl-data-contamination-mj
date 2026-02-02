"""
Runtime patches for vLLM/HF tqdm interop.
This module is auto-imported by Python when on PYTHONPATH.
"""

def _patch_vllm_disabled_tqdm():
    try:
        from vllm.model_executor.model_loader import weight_utils  # type: ignore
    except Exception:
        return

    if not hasattr(weight_utils, "DisabledTqdm"):
        return

    original_init = weight_utils.DisabledTqdm.__init__

    def _wrapped_init(self, *args, **kwargs):
        kwargs.pop("disable", None)
        return original_init(self, *args, **kwargs)

    weight_utils.DisabledTqdm.__init__ = _wrapped_init


_patch_vllm_disabled_tqdm()
