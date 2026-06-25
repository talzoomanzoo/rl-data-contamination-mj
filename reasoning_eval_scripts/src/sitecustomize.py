"""
Runtime patches for vLLM/HF tqdm interop.
This module is auto-imported by Python when on PYTHONPATH.
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from tokenizer_extended_compat import apply_transformers_special_tokens_extended_getattr_compat

    apply_transformers_special_tokens_extended_getattr_compat()
except ImportError:
    pass


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


def _patch_vllm_get_cached_tokenizer_safe():
    mod = None
    try:
        from vllm.tokenizers import hf as _hf_mod  # type: ignore

        if getattr(_hf_mod, "get_cached_tokenizer", None) is not None:
            mod = _hf_mod
    except Exception:
        pass
    if mod is None:
        try:
            from vllm.transformers_utils import tokenizer as _legacy_mod  # type: ignore

            if getattr(_legacy_mod, "get_cached_tokenizer", None) is not None:
                mod = _legacy_mod
        except Exception:
            pass
    if mod is None:
        return

    _orig = mod.get_cached_tokenizer

    def _get_cached_tokenizer_safe(tokenizer):
        if not hasattr(tokenizer, "all_special_tokens_extended"):
            tokenizer.all_special_tokens_extended = list(tokenizer.all_special_tokens)
        return _orig(tokenizer)

    mod.get_cached_tokenizer = _get_cached_tokenizer_safe


_patch_vllm_disabled_tqdm()
_patch_vllm_get_cached_tokenizer_safe()
