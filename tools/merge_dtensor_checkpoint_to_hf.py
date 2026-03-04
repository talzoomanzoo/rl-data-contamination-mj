#!/usr/bin/env python3
"""
Merge a DTensor-sharded checkpoint (saved as model_world_size_{N}_rank_{r}.pt)
into a standard Hugging Face model directory that vLLM/Transformers can load.

This is intended for checkpoints that look like:
  - model_world_size_8_rank_0.pt ... model_world_size_8_rank_7.pt
  - huggingface/config.json, tokenizer.json, tokenizer_config.json, etc.

The per-rank files contain DTensors sharded along dim=0; we reconstruct each
parameter by concatenating local shards along dim=0 in rank order.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List, Optional, Tuple


def _maybe_snapshot_download(repo_id: str, revision: Optional[str]) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        # Keep it broad; the repo is small already (~31 files).
    )


def _read_world_size_from_fsdp_config(ckpt_dir: str) -> Optional[int]:
    import json

    p = os.path.join(ckpt_dir, "fsdp_config.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ws = cfg.get("world_size")
    return int(ws) if ws is not None else None


def _copy_hf_metadata(ckpt_dir: str, out_dir: str) -> None:
    hf_dir = os.path.join(ckpt_dir, "huggingface")
    if not os.path.isdir(hf_dir):
        raise FileNotFoundError(f"Missing huggingface metadata dir: {hf_dir}")

    os.makedirs(out_dir, exist_ok=True)

    for name in os.listdir(hf_dir):
        src = os.path.join(hf_dir, name)
        dst = os.path.join(out_dir, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _load_rank_state(path: str):
    import torch
    import torch.distributed.tensor  # noqa: F401

    # NOTE: weights_only=False is required for DTensor objects in recent PyTorch.
    return torch.load(path, map_location="cpu", weights_only=False)


def _is_dtensor(x) -> bool:
    return x.__class__.__module__.startswith("torch.distributed.tensor") and x.__class__.__name__ == "DTensor"


def _merge_shards(
    ckpt_dir: str,
    world_size: int,
    filename_template: str,
    strict_keys: bool,
) -> Dict[str, "torch.Tensor"]:
    import torch

    shard_parts: Dict[str, List[Optional[torch.Tensor]]] = {}
    shard_meta: Dict[str, Tuple[bool, Tuple[int, ...]]] = {}
    expected_keys: Optional[List[str]] = None

    for rank in range(world_size):
        rank_path = os.path.join(ckpt_dir, filename_template.format(world_size=world_size, rank=rank))
        if not os.path.exists(rank_path):
            raise FileNotFoundError(f"Missing rank shard: {rank_path}")

        state = _load_rank_state(rank_path)
        if expected_keys is None:
            expected_keys = list(state.keys())
        elif strict_keys and list(state.keys()) != expected_keys:
            raise ValueError(
                "Shard key mismatch between ranks. "
                "Set --no_strict_keys to allow union merge, but results may be wrong."
            )

        for k, v in state.items():
            if _is_dtensor(v):
                local = v.to_local().contiguous()
                is_dt = True
                global_shape = tuple(int(s) for s in v.shape)
            elif isinstance(v, torch.Tensor):
                local = v.contiguous()
                is_dt = False
                global_shape = tuple(int(s) for s in v.shape)
            else:
                raise TypeError(f"Unsupported value type for key {k}: {type(v)}")

            parts = shard_parts.setdefault(k, [None] * world_size)
            parts[rank] = local

            # Record meta from first observation.
            if k not in shard_meta:
                shard_meta[k] = (is_dt, global_shape)

        # Drop state dict aggressively to keep peak memory down.
        del state

    merged: Dict[str, torch.Tensor] = {}
    for k, parts in shard_parts.items():
        is_dt, global_shape = shard_meta[k]
        if any(p is None for p in parts):
            missing = [i for i, p in enumerate(parts) if p is None]
            raise ValueError(f"Missing shards for key {k}: ranks={missing}")

        if is_dt:
            full = torch.cat(parts, dim=0)
            if tuple(full.shape) != global_shape:
                raise ValueError(f"Shape mismatch for {k}: got {tuple(full.shape)} expected {global_shape}")
            merged[k] = full
        else:
            # Replicated parameter: take rank0; optionally validate equality.
            merged[k] = parts[0]

    return merged


def _save_state_dict(out_dir: str, state_dict) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # Prefer safetensors when available.
    try:
        from safetensors.torch import save_file as safe_save_file

        out_path = os.path.join(out_dir, "model.safetensors")
        safe_save_file(state_dict, out_path)
        return out_path
    except Exception:
        import torch

        out_path = os.path.join(out_dir, "pytorch_model.bin")
        torch.save(state_dict, out_path)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Either a local checkpoint directory, or a Hugging Face repo id.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to write the merged Hugging Face model folder.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch/tag/commit) when --checkpoint is a repo id.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Number of rank shards. If omitted, read from fsdp_config.json.",
    )
    parser.add_argument(
        "--filename_template",
        default="model_world_size_{world_size}_rank_{rank}.pt",
        help="Filename template inside checkpoint dir.",
    )
    parser.add_argument(
        "--no_strict_keys",
        action="store_true",
        help="Allow union of keys across ranks (not recommended).",
    )
    args = parser.parse_args()

    if os.path.isdir(args.checkpoint):
        ckpt_dir = os.path.abspath(args.checkpoint)
    else:
        ckpt_dir = _maybe_snapshot_download(args.checkpoint, args.revision)

    world_size = args.world_size or _read_world_size_from_fsdp_config(ckpt_dir)
    if not world_size:
        raise ValueError("Could not determine world_size. Pass --world_size explicitly.")

    print(f"[merge] checkpoint_dir={ckpt_dir}")
    print(f"[merge] world_size={world_size}")
    print(f"[merge] output_dir={os.path.abspath(args.output_dir)}")

    merged = _merge_shards(
        ckpt_dir=ckpt_dir,
        world_size=world_size,
        filename_template=args.filename_template,
        strict_keys=not args.no_strict_keys,
    )

    _copy_hf_metadata(ckpt_dir, args.output_dir)
    out_weights = _save_state_dict(args.output_dir, merged)
    print(f"[merge] wrote weights: {out_weights}")
    print("[merge] done.")


if __name__ == "__main__":
    main()

