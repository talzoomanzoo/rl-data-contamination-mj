#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

import tqdm.asyncio as _tqdm_asyncio

from kk_utils import batch_decode_vllm

def _extract_messages(prompt_field):
    if isinstance(prompt_field, list) and prompt_field:
        if isinstance(prompt_field[0], dict):
            messages = []
            for msg in prompt_field:
                messages.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    }
                )
            return messages
    return [{"role": "user", "content": str(prompt_field)}]


def _messages_to_text(messages):
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts)


def _build_eval_prompt(problem_text, ground_truth, k_vars=None):
    k_clause = ""
    if k_vars is not None:
        k_clause = f"The solution should be a length-{k_vars} binary string.\n\n"
    return (
        "You are evaluating a SAT solution.\n\n"
        "SAT problem:\n"
        f"{problem_text}\n\n"
        "Model response:\n"
        "<MODEL_RESPONSE>\n\n"
        f"{k_clause}"
        "Ground-truth solution string:\n"
        f"{ground_truth}\n\n"
        "Extract the final answer from the model response (prefer text inside "
        "\\boxed{}, otherwise the last contiguous 0/1 string). "
        "Compare it to the ground-truth solution. "
        "Return a JSON object with keys: extracted_answer, is_correct, reason."
    )


def _build_generation_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return _messages_to_text(messages)


def _ensure_tokenizer_compat():
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        @property
        def all_special_tokens_extended(self):
            return self.all_special_tokens
        PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


def _patch_tqdm_disable():
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


def _extract_binary_answer(response_text):
    if not response_text:
        return None
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response_text)
    if boxed:
        return boxed[-1].strip()
    tail = re.findall(r"[01]+", response_text)
    if tail:
        return tail[-1].strip()
    return None


def _evaluate_response(response_text, ground_truth):
    extracted = _extract_binary_answer(response_text)
    if extracted is None or ground_truth is None:
        return {
            "extracted_answer": extracted,
            "is_correct": False,
            "reason": "missing_extracted_or_ground_truth",
        }
    is_correct = str(extracted) == str(ground_truth)
    return {
        "extracted_answer": extracted,
        "is_correct": is_correct,
        "reason": "match" if is_correct else "mismatch",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate SAT responses, evaluations, and accuracy."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to SAT parquet dataset.",
    )
    parser.add_argument(
        "--gen_output",
        type=str,
        default=None,
        help="Output JSON file for generated responses.",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Output JSON file for evaluated responses.",
    )
    parser.add_argument(
        "--acc_output",
        type=str,
        default=None,
        help="Output JSON file for final accuracy.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM with batched decoding.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for vLLM decoding.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for distributed vLLM inference.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Optional max model length for vLLM.",
    )
    parser.add_argument(
        "--rope_scaling_type",
        type=str,
        default="dynamic",
        choices=["dynamic", "linear", "none"],
        help="vLLM rope scaling type (use 'none' to disable).",
    )
    parser.add_argument(
        "--rope_scaling_factor",
        type=float,
        default=1.0,
        help="vLLM rope scaling factor.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 = greedy).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling (default: greedy).",
    )

    args = parser.parse_args()

    df = pd.read_parquet(args.dataset_path)
    start_idx = max(args.start_idx, 0)
    end_idx = len(df) if args.limit is None else min(len(df), start_idx + args.limit)
    df = df.iloc[start_idx:end_idx]

    dataset_tag = Path(args.dataset_path).stem
    model_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model).strip("_")
    base_dir = Path(args.gen_output or args.eval_output or ".")
    base_dir = base_dir.parent if base_dir.suffix else base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.gen_output is None:
        args.gen_output = str(base_dir / f"{dataset_tag}__{model_tag}__generations.json")
    if args.eval_output is None:
        args.eval_output = str(base_dir / f"{dataset_tag}__{model_tag}__evaluated.json")
    if args.acc_output is None:
        args.acc_output = str(base_dir / f"{dataset_tag}__{model_tag}__accuracy.json")

    _ensure_tokenizer_compat()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.use_vllm:
        from vllm import LLM  # type: ignore

        class _SimpleLLM:
            def __init__(self, model, max_tokens):
                self.model = model
                self.max_tokens = max_tokens

        _patch_tqdm_disable()
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("VLLM_DISABLE_TQDM", "1")

        rope_scaling = None
        if args.rope_scaling_type != "none":
            rope_scaling = {
                "type": args.rope_scaling_type,
                "factor": args.rope_scaling_factor,
            }
        vllm_kwargs = {
            "model": args.model,
            "tokenizer": args.model,
            "trust_remote_code": True,
            "tokenizer_mode": "auto",
            "rope_scaling": rope_scaling,
            "tensor_parallel_size": args.tensor_parallel_size,
        }
        if args.max_model_len is not None:
            vllm_kwargs["max_model_len"] = args.max_model_len
        vllm_model = LLM(**vllm_kwargs)
        llm = _SimpleLLM(vllm_model, args.max_new_tokens)
        model = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        model.eval()
        llm = None

    total = 0
    correct_count = 0

    def _write_json_item(fp, item, first_flag):
        if not first_flag[0]:
            fp.write(",\n")
        fp.write(json.dumps(item, ensure_ascii=False))
        first_flag[0] = False

    def _generate_batch(batch_rows, batch_prompts):
        nonlocal total, correct_count
        if not batch_rows:
            return
        if args.use_vllm:
            responses = batch_decode_vllm(
                llm, batch_prompts, batch_size=args.batch_size, use_tqdm=False
            )
        else:
            responses = []
            for gen_prompt in batch_prompts:
                inputs = tokenizer(gen_prompt, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature if args.do_sample else None,
                        top_p=args.top_p if args.do_sample else None,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                responses.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

        for row_data, response_text in zip(batch_rows, responses):
            gen_record = {
                "prompt": row_data["prompt_text"],
                "messages": row_data["messages"],
                "ground_truth": row_data["ground_truth"],
                "response": response_text,
                "meta": row_data["meta"],
            }
            eval_result = _evaluate_response(response_text, row_data["ground_truth"])
            if eval_result["is_correct"]:
                correct_count += 1
            eval_record = {
                **gen_record,
                "eval_prompt": row_data["eval_prompt"],
                "eval": eval_result,
            }
            _write_json_item(gen_f, gen_record, gen_first)
            _write_json_item(eval_f, eval_record, eval_first)
            total += 1

    with open(args.gen_output, "w", encoding="utf-8") as gen_f, open(
        args.eval_output, "w", encoding="utf-8"
    ) as eval_f:
        gen_f.write("[\n")
        eval_f.write("[\n")
        gen_first = [True]
        eval_first = [True]

        batch_rows = []
        batch_prompts = []
        for row_idx, row in df.iterrows():
            prompt_field = row.get("prompt")
            messages = _extract_messages(prompt_field)
            prompt_text = _messages_to_text(messages)
            gen_prompt = _build_generation_prompt(tokenizer, messages)

            reward_model = row.get("reward_model") or {}
            ground_truth = reward_model.get("ground_truth")
            extra_info = row.get("extra_info") or {}
            k_vars = extra_info.get("k_vars")

            batch_rows.append(
                {
                    "prompt_text": prompt_text,
                    "messages": messages,
                    "ground_truth": ground_truth,
                    "eval_prompt": _build_eval_prompt(prompt_text, ground_truth, k_vars),
                    "meta": {
                        "data_source": row.get("data_source"),
                        "member": row.get("member"),
                        "extra_info": extra_info,
                    },
                }
            )
            batch_prompts.append(gen_prompt)

            if len(batch_rows) >= args.batch_size:
                _generate_batch(batch_rows, batch_prompts)
                batch_rows = []
                batch_prompts = []

        _generate_batch(batch_rows, batch_prompts)

        gen_f.write("\n]\n")
        eval_f.write("\n]\n")

    total = total
    accuracy = (correct_count / total) if total else 0.0
    acc_record = {
        "total": total,
        "correct": correct_count,
        "accuracy": round(accuracy, 6),
        "model": args.model,
        "dataset_path": args.dataset_path,
        "gen_output": args.gen_output,
        "eval_output": args.eval_output,
    }
    with open(args.acc_output, "w", encoding="utf-8") as f:
        json.dump(acc_record, f, ensure_ascii=False, indent=2)

    print(f"Wrote generations to {args.gen_output}")
    print(f"Wrote evaluations to {args.eval_output}")
    print(f"Wrote accuracy to {args.acc_output}")


if __name__ == "__main__":
    main()