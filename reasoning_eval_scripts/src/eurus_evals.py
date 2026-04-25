#!/usr/bin/env python3
import argparse
import asyncio
import glob
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _ensure_tokenizer_compat():
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        @property
        def all_special_tokens_extended(self):  # type: ignore
            return self.all_special_tokens

        PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended  # type: ignore


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


def _disable_hf_transfer_if_missing():
    # Some environments set HF_HUB_ENABLE_HF_TRANSFER=1 globally, but don't have
    # `hf_transfer` installed. That breaks model downloads.
    try:
        from huggingface_hub import constants as _hf_constants  # type: ignore
    except Exception:
        _hf_constants = None

    enabled = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1"
    if _hf_constants is not None:
        enabled = enabled or bool(getattr(_hf_constants, "HF_HUB_ENABLE_HF_TRANSFER", False))
    if not enabled:
        return
    try:
        import hf_transfer  # type: ignore  # noqa: F401
    except Exception:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        if _hf_constants is not None and hasattr(_hf_constants, "HF_HUB_ENABLE_HF_TRANSFER"):
            _hf_constants.HF_HUB_ENABLE_HF_TRANSFER = False  # type: ignore
        try:
            from huggingface_hub import file_download as _hf_file_download  # type: ignore

            if hasattr(_hf_file_download, "HF_HUB_ENABLE_HF_TRANSFER"):
                _hf_file_download.HF_HUB_ENABLE_HF_TRANSFER = False  # type: ignore
        except Exception:
            pass


def _to_py(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _extract_messages(prompt_field: Any) -> List[Dict[str, str]]:
    prompt_field = _to_py(prompt_field)
    if isinstance(prompt_field, list) and prompt_field:
        if isinstance(prompt_field[0], dict):
            return [
                {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                for m in prompt_field
            ]
    return [{"role": "user", "content": str(prompt_field)}]


def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    return "\n".join([f"{m.get('role','user').upper()}: {m.get('content','')}" for m in messages])


def _build_generation_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return _messages_to_text(messages)


def _prompt_key(prompt_field: Any) -> str:
    prompt_field = _to_py(prompt_field)
    dumped = json.dumps(prompt_field, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()


def _find_repo_root(dataset_path: str) -> Path:
    candidates: List[Path] = []
    try:
        candidates.append(Path(dataset_path).expanduser().resolve())
    except Exception:
        pass
    candidates.append(Path(__file__).resolve())
    candidates.append(Path.cwd().resolve())

    for base in candidates:
        for parent in [base, *base.parents]:
            if (parent / ".git").exists():
                return parent
        for parent in [base, *base.parents]:
            if parent.name == "benchmarks":
                continue
            if (parent / "benchmarks" / "filtered_samples").is_dir() or (parent / "benchmarks" / "EURUS").is_dir():
                return parent
    return Path.cwd().resolve()


def _candidate_answer_parquets(repo_root: Path) -> List[str]:
    roots = [
        repo_root / "benchmarks" / "filtered_samples",
        repo_root / "benchmarks" / "not_used",
        repo_root / "benchmarks",
    ]
    paths: List[str] = []
    for r in roots:
        paths.extend(glob.glob(str(Path(r) / "**" / "*.parquet"), recursive=True))
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _build_answer_lookup(parquet_paths: Iterable[str]) -> Dict[Tuple[Optional[str], str], Any]:
    lookup: Dict[Tuple[Optional[str], str], Any] = {}
    for p in parquet_paths:
        try:
            df = pd.read_parquet(p, engine="pyarrow")
        except Exception:
            continue
        if "prompt" not in df.columns or "answer" not in df.columns:
            continue
        has_ds = "data_source" in df.columns
        for _, row in df.iterrows():
            ds = str(row["data_source"]) if has_ds and row.get("data_source") is not None else None
            key = _prompt_key(row["prompt"])
            val = row.get("answer")
            lookup.setdefault((ds, key), val)
            lookup.setdefault((None, key), val)
    return lookup


def _ensure_answers(df: pd.DataFrame, dataset_path: str) -> pd.DataFrame:
    if "answer" in df.columns and df["answer"].notna().any():
        return df

    repo_root = _find_repo_root(dataset_path)
    candidates = _candidate_answer_parquets(repo_root)
    lookup = _build_answer_lookup(candidates)

    answers: List[Any] = []
    missing = 0
    for _, row in df.iterrows():
        ds = str(row["data_source"]) if "data_source" in df.columns and row.get("data_source") is not None else None
        key = _prompt_key(row["prompt"])
        ans = lookup.get((ds, key)) or lookup.get((None, key))
        if ans is None:
            missing += 1
        answers.append(ans)

    df = df.copy()
    df["answer"] = answers
    if missing:
        raise SystemExit(
            f"Could not resolve {missing}/{len(df)} answers for dataset {dataset_path}. "
            "Provide a parquet with an `answer` column, or add a matching parquet under benchmarks/."
        )
    return df


def _extract_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    return m[-1].strip() if m else None


def _extract_last_numberish(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.findall(r"(-?\d+(?:/\d+)?(?:\.\d+)?)", text)
    return m[-1].strip() if m else None


def extract_answer_from_response(response_text: str) -> Optional[str]:
    return _extract_boxed(response_text) or _extract_last_numberish(response_text)


def normalize_answer(answer: Any) -> Optional[str]:
    if answer is None:
        return None
    s = str(answer).strip()
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]+)\}", r"\1", s)
    s = re.sub(r"[^0-9a-zA-Z/.-]+", "", s)
    return s.lower() if s else None


def evaluate_response(response_text: str, ground_truth_answer: Any) -> Dict[str, Any]:
    extracted = extract_answer_from_response(response_text)
    ne = normalize_answer(extracted)
    ng = normalize_answer(ground_truth_answer)
    ok = (ne is not None) and (ng is not None) and (ne == ng)
    return {
        "extracted_answer": extracted,
        "ground_truth": None if ground_truth_answer is None else str(ground_truth_answer),
        "is_correct": bool(ok),
        "reason": "match" if ok else ("missing_extracted" if extracted is None else "mismatch"),
    }


def _sanitize_model_tag(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model).strip("_")


def _answer_target_text(answer: Any) -> str:
    return "\\boxed{" + str(answer).strip() + "}"


def _as_member_flag(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(int(value))
    except Exception:
        return None


def _none_if_nan(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return None if np.isnan(x) else float(x)
    except Exception:
        return float(x)


def _compute_answer_logp_transformers(*, model, tokenizer, prompt_text: str, answer_target: str) -> float:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt_text + answer_target, add_special_tokens=False).input_ids
    ans_ids = full_ids[len(prompt_ids) :]
    if not ans_ids:
        return 0.0

    input_ids = torch.tensor([full_ids], device=model.device, dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids).logits
        lp = torch.log_softmax(logits, dim=-1)
    start = len(prompt_ids)
    total = 0.0
    for pos, tok in enumerate(ans_ids, start=start):
        if pos == 0:
            continue
        total += float(lp[0, pos - 1, tok].item())
    return total


def _compute_answer_logp_vllm_sync(
    *, vllm_model, tokenizer, prompt_text: str, answer_target: str, prompt_logprobs_k: int
) -> float:
    from vllm import SamplingParams  # type: ignore

    full_text = prompt_text + answer_target
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    ans_ids = full_ids[len(prompt_ids) :]
    if not ans_ids:
        return 0.0

    outs = vllm_model.generate(
        [full_text], SamplingParams(max_tokens=1, prompt_logprobs=int(prompt_logprobs_k)), use_tqdm=False
    )
    if not outs:
        return float("nan")
    plp = getattr(outs[0], "prompt_logprobs", None)
    if plp is None:
        return float("nan")

    start = len(prompt_ids)
    total = 0.0
    for pos, tok in enumerate(ans_ids, start=start):
        if pos >= len(plp) or plp[pos] is None:
            return float("nan")
        entry = plp[pos].get(int(tok))
        if entry is None:
            return float("nan")
        total += float(entry.logprob)
    return total


async def _init_vllm_async_engine(
    *,
    model: str,
    tensor_parallel_size: int,
    max_model_len: Optional[int],
    rope_scaling_type: str,
    rope_scaling_factor: float,
    gpu_memory_utilization: float,
    max_num_seqs: int,
):
    from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
    from vllm import AsyncLLMEngine  # type: ignore

    rope_scaling = None
    if rope_scaling_type != "none":
        rope_scaling = {"type": rope_scaling_type, "factor": rope_scaling_factor}

    engine_args = AsyncEngineArgs(
        model=model,
        tokenizer=model,
        trust_remote_code=True,
        tokenizer_mode="auto",
        tensor_parallel_size=tensor_parallel_size,
        rope_scaling=rope_scaling,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


async def _generate_vllm_async_engine(
    engine,
    prompts: List[str],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
    max_in_flight: int,
    pbar=None,
) -> List[List[str]]:
    from vllm import SamplingParams  # type: ignore

    params = SamplingParams(
        n=int(num_samples),
        max_tokens=max_new_tokens,
        temperature=(temperature if do_sample else 0.0),
        top_p=(top_p if do_sample else 1.0),
        repetition_penalty=float(repetition_penalty),
    )

    sem = asyncio.Semaphore(max(1, int(max_in_flight)))
    results: List[Optional[List[str]]] = [None] * len(prompts)

    async def _one(i: int, prompt: str) -> Tuple[int, List[str]]:
        async with sem:
            final = None
            try:
                async for out in engine.generate(prompt, params, request_id=f"gen-{i}"):
                    final = out
            except Exception:
                final = None
            if final is None or not getattr(final, "outputs", None):
                return i, ["" for _ in range(num_samples)]
            texts = [o.text for o in final.outputs]
            if len(texts) < num_samples:
                texts = texts + [""] * (num_samples - len(texts))
            return i, texts[:num_samples]

    tasks = [asyncio.create_task(_one(i, p)) for i, p in enumerate(prompts)]
    for fut in asyncio.as_completed(tasks):
        i, texts = await fut
        results[i] = texts
        if pbar is not None:
            pbar.update(1)

    return [r if r is not None else ["" for _ in range(num_samples)] for r in results]


async def _score_answer_logp_vllm_async_engine(
    engine,
    *,
    tokenizer,
    prompt_texts: List[str],
    answer_targets: List[str],
    max_in_flight: int,
    prompt_logprobs_k: int,
) -> List[float]:
    from vllm import SamplingParams  # type: ignore

    sem = asyncio.Semaphore(max(1, int(max_in_flight)))
    results: List[Optional[float]] = [None] * len(prompt_texts)

    async def _one(i: int, prompt_text: str, answer_target: str) -> Tuple[int, float]:
        async with sem:
            try:
                full_text = prompt_text + answer_target
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
                full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
                ans_ids = full_ids[len(prompt_ids) :]
                if not ans_ids:
                    return i, 0.0

                params = SamplingParams(max_tokens=1, prompt_logprobs=int(prompt_logprobs_k))
                final = None
                async for out in engine.generate(full_text, params, request_id=f"score-{i}"):
                    final = out
                if final is None or getattr(final, "prompt_logprobs", None) is None:
                    return i, float("nan")

                plp = final.prompt_logprobs
                start = len(prompt_ids)
                total = 0.0
                for pos, tok in enumerate(ans_ids, start=start):
                    if pos >= len(plp) or plp[pos] is None:
                        return i, float("nan")
                    entry = plp[pos].get(int(tok))
                    if entry is None:
                        return i, float("nan")
                    total += float(entry.logprob)
                return i, total
            except Exception:
                return i, float("nan")

    tasks = [
        asyncio.create_task(_one(i, p, a))
        for i, (p, a) in enumerate(zip(prompt_texts, answer_targets))
    ]
    for fut in asyncio.as_completed(tasks):
        i, val = await fut
        results[i] = val
    return [r if r is not None else float("nan") for r in results]


def main():
    parser = argparse.ArgumentParser(description="Generate + evaluate EURUS-style prompt/answer parquet datasets.")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name or path.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to parquet dataset.")
    parser.add_argument("--gen_output", type=str, default=None, help="Output JSON file for generations.")
    parser.add_argument("--eval_output", type=str, default=None, help="Output JSON file for evaluations.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (default: 0).")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows.")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0=greedy).")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of sampled responses per prompt (acc@k).")
    parser.add_argument(
        "--prompt_logprobs_k",
        type=int,
        default=50,
        help="How many prompt logprobs to request from vLLM when computing log p(answer|prompt).",
    )

    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM.")
    parser.add_argument("--async_vllm", action="store_true", help="Use vLLM AsyncLLMEngine (continuous batching).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for sync generation.")
    parser.add_argument("--max_in_flight", type=int, default=None, help="Max concurrent in-flight async requests.")
    parser.add_argument(
        "--score_max_in_flight",
        type=int,
        default=None,
        help="Max concurrent in-flight async logp scoring requests.",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="vLLM gpu_memory_utilization.")
    parser.add_argument("--max_num_seqs", type=int, default=64, help="vLLM max_num_seqs.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor_parallel_size.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM max_model_len.")
    parser.add_argument(
        "--rope_scaling_type",
        type=str,
        default="dynamic",
        choices=["dynamic", "linear", "none"],
        help="vLLM rope scaling type.",
    )
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0, help="vLLM rope scaling factor.")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset_path, engine="pyarrow")
    if "prompt" not in df.columns:
        raise SystemExit(f"Dataset {args.dataset_path} must contain `prompt`. Found: {list(df.columns)}")
    df = _ensure_answers(df, args.dataset_path)
    start_idx = max(args.start_idx, 0)
    end_idx = len(df) if args.limit is None else min(len(df), start_idx + args.limit)
    df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    dataset_tag = Path(args.dataset_path).stem
    model_tag = _sanitize_model_tag(args.model)
    base_dir = Path(args.gen_output or args.eval_output or ".")
    base_dir = base_dir.parent if base_dir.suffix else base_dir
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise SystemExit(
            f"Cannot create output directory: {base_dir} ({e}). "
            "Choose a writable path for --gen_output/--eval_output (e.g. ./data/...)."
        )
    if args.gen_output is None:
        args.gen_output = str(base_dir / f"{dataset_tag}__{model_tag}__generations.json")
    if args.eval_output is None:
        args.eval_output = str(base_dir / f"{dataset_tag}__{model_tag}__evaluated.json")

    _disable_hf_transfer_if_missing()
    _ensure_tokenizer_compat()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Model backends
    llm = None
    hf_model = None
    if args.use_vllm and not args.async_vllm:
        from vllm import LLM  # type: ignore

        _patch_tqdm_disable()
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("VLLM_DISABLE_TQDM", "1")

        rope_scaling = None
        if args.rope_scaling_type != "none":
            rope_scaling = {"type": args.rope_scaling_type, "factor": args.rope_scaling_factor}
        vllm_kwargs = {
            "model": args.model,
            "tokenizer": args.model,
            "trust_remote_code": True,
            "tokenizer_mode": "auto",
            "rope_scaling": rope_scaling,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
        }
        if args.max_model_len is not None:
            vllm_kwargs["max_model_len"] = args.max_model_len

        vllm_model = LLM(**vllm_kwargs)

        class _SimpleLLM:
            def __init__(self, model, max_tokens):
                self.model = model
                self.max_tokens = max_tokens

        llm = _SimpleLLM(vllm_model, args.max_new_tokens)
    elif not args.use_vllm:
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
            )
        except ValueError as e:
            if "requires `accelerate`" not in str(e):
                raise
            hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", trust_remote_code=True)
            if torch.cuda.is_available():
                hf_model = hf_model.to("cuda")
        hf_model.eval()

    # Stats accumulators
    total = 0
    correct_at_k = 0
    member_total = 0
    member_correct_at_k = 0
    nonmember_total = 0
    nonmember_correct_at_k = 0
    member_logp: List[float] = []
    nonmember_logp: List[float] = []

    pbar = tqdm(total=len(df), desc=f"Processed ({model_tag})", unit="ex", dynamic_ncols=True) if tqdm else None

    def _write_json_item(fp, item, first_flag):
        if not first_flag[0]:
            fp.write(",\n")
        fp.write(json.dumps(item, ensure_ascii=False))
        first_flag[0] = False

    def _process_one(row_data: Dict[str, Any], response_texts: List[str], logp_val: Optional[float], answer_target: str):
        nonlocal total, correct_at_k, member_total, member_correct_at_k, nonmember_total, nonmember_correct_at_k
        gen_record = {
            "messages": row_data["messages"],
            "prompt_text": row_data["prompt_text"],
            "answer": row_data["answer"],
            "responses": response_texts,
            "meta": row_data["meta"],
        }
        per_sample = [evaluate_response(r, row_data["answer"]) for r in response_texts]
        is_correct_k = any(r.get("is_correct") for r in per_sample)
        if is_correct_k:
            correct_at_k += 1

        mem_flag = _as_member_flag(row_data["meta"].get("member"))
        if logp_val is not None:
            if mem_flag is True:
                member_logp.append(float(logp_val))
            elif mem_flag is False:
                nonmember_logp.append(float(logp_val))
        if mem_flag is True:
            member_total += 1
            if is_correct_k:
                member_correct_at_k += 1
        elif mem_flag is False:
            nonmember_total += 1
            if is_correct_k:
                nonmember_correct_at_k += 1

        eval_record = {
            **gen_record,
            "evals": per_sample,
            "acc_at_k": bool(is_correct_k),
            "k": int(args.num_samples),
            "answer_target": answer_target,
            "logp_answer_given_prompt": _none_if_nan(logp_val),
        }
        _write_json_item(gen_f, gen_record, gen_first)
        _write_json_item(eval_f, eval_record, eval_first)
        total += 1

    with open(args.gen_output, "w", encoding="utf-8") as gen_f, open(args.eval_output, "w", encoding="utf-8") as eval_f:
        gen_f.write("[\n")
        eval_f.write("[\n")
        gen_first = [True]
        eval_first = [True]

        if args.use_vllm and args.async_vllm:
            # Build all prompts first
            rows: List[Dict[str, Any]] = []
            prompts: List[str] = []
            answer_targets: List[str] = []
            member_flags: List[Optional[bool]] = []
            for _, row in df.iterrows():
                messages = _extract_messages(row.get("prompt"))
                prompt_text = _messages_to_text(messages)
                gen_prompt = _build_generation_prompt(tokenizer, messages)
                rows.append(
                    {
                        "messages": messages,
                        "prompt_text": prompt_text,
                        "answer": row.get("answer"),
                        "meta": {"data_source": row.get("data_source"), "member": row.get("member"), "gen_prompt": gen_prompt},
                    }
                )
                prompts.append(gen_prompt)
                answer_targets.append(_answer_target_text(row.get("answer")))
                member_flags.append(_as_member_flag(row.get("member")))

            max_in_flight = args.max_in_flight if args.max_in_flight is not None else min(args.batch_size, 8)
            score_max_in_flight = (
                args.score_max_in_flight
                if args.score_max_in_flight is not None
                else max(1, min(2, int(max_in_flight)))
            )

            async def _run_async():
                engine = await _init_vllm_async_engine(
                    model=args.model,
                    tensor_parallel_size=args.tensor_parallel_size,
                    max_model_len=args.max_model_len,
                    rope_scaling_type=args.rope_scaling_type,
                    rope_scaling_factor=args.rope_scaling_factor,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_num_seqs=args.max_num_seqs,
                )
                try:
                    all_responses = await _generate_vllm_async_engine(
                        engine,
                        prompts,
                        num_samples=args.num_samples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=args.do_sample,
                        max_in_flight=max_in_flight,
                        pbar=pbar,
                    )
                    logps = await _score_answer_logp_vllm_async_engine(
                        engine,
                        tokenizer=tokenizer,
                        prompt_texts=prompts,
                        answer_targets=answer_targets,
                        max_in_flight=score_max_in_flight,
                        prompt_logprobs_k=args.prompt_logprobs_k,
                    )
                    return all_responses, logps
                finally:
                    try:
                        await engine.shutdown()  # type: ignore[attr-defined]
                    except Exception:
                        pass

            all_responses, logps = asyncio.run(_run_async())
            for row_data, response_texts, logp_val, mem_flag, answer_target in zip(
                rows, all_responses, logps, member_flags, answer_targets
            ):
                if pbar is not None:
                    # pbar is updated during async generation; do not update here.
                    pass
                _process_one(row_data, response_texts, float(logp_val), answer_target)
        else:
            batch_rows: List[Dict[str, Any]] = []
            batch_prompts: List[str] = []
            for _, row in df.iterrows():
                messages = _extract_messages(row.get("prompt"))
                prompt_text = _messages_to_text(messages)
                gen_prompt = _build_generation_prompt(tokenizer, messages)
                batch_rows.append(
                    {
                        "messages": messages,
                        "prompt_text": prompt_text,
                        "answer": row.get("answer"),
                        "meta": {"data_source": row.get("data_source"), "member": row.get("member"), "gen_prompt": gen_prompt},
                    }
                )
                batch_prompts.append(gen_prompt)

                if len(batch_rows) >= args.batch_size:
                    # generate
                    if args.use_vllm:
                        from vllm import SamplingParams  # type: ignore

                        sp = SamplingParams(
                            n=int(args.num_samples),
                            max_tokens=args.max_new_tokens,
                            temperature=(args.temperature if args.do_sample else 0.0),
                            top_p=(args.top_p if args.do_sample else 1.0),
                            repetition_penalty=float(args.repetition_penalty),
                        )
                        outs = llm.model.generate(batch_prompts, sp, use_tqdm=False)
                        responses = []
                        for out in outs:
                            texts = [o.text for o in out.outputs]
                            if len(texts) < args.num_samples:
                                texts = texts + [""] * (args.num_samples - len(texts))
                            responses.append(texts[: args.num_samples])
                    else:
                        responses = []
                        for gp in batch_prompts:
                            inputs = tokenizer(gp, return_tensors="pt").to(hf_model.device)
                            with torch.inference_mode():
                                output_ids = hf_model.generate(
                                    **inputs,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=args.do_sample,
                                    temperature=args.temperature if args.do_sample else None,
                                    top_p=args.top_p if args.do_sample else None,
                                    pad_token_id=tokenizer.eos_token_id,
                                    repetition_penalty=float(args.repetition_penalty),
                                    num_return_sequences=int(args.num_samples),
                                )
                            decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output_ids]
                            responses.append(decoded)

                    for row_data, resp_texts in zip(batch_rows, responses):
                        if pbar is not None:
                            pbar.update(1)
                        answer_target = _answer_target_text(row_data["answer"])
                        logp_val = None
                        if args.use_vllm:
                            logp_val = _compute_answer_logp_vllm_sync(
                                vllm_model=llm.model,
                                tokenizer=tokenizer,
                                prompt_text=row_data["meta"]["gen_prompt"],
                                answer_target=answer_target,
                                prompt_logprobs_k=args.prompt_logprobs_k,
                            )
                        else:
                            logp_val = _compute_answer_logp_transformers(
                                model=hf_model,
                                tokenizer=tokenizer,
                                prompt_text=row_data["meta"]["gen_prompt"],
                                answer_target=answer_target,
                            )
                        _process_one(row_data, resp_texts, logp_val, answer_target)

                    batch_rows = []
                    batch_prompts = []

            if batch_rows:
                # flush remaining by recursion: simplest reuse by setting batch_size large
                # (copy logic inline to avoid duplicating too much)
                if args.use_vllm:
                    from vllm import SamplingParams  # type: ignore

                    sp = SamplingParams(
                        n=int(args.num_samples),
                        max_tokens=args.max_new_tokens,
                        temperature=(args.temperature if args.do_sample else 0.0),
                        top_p=(args.top_p if args.do_sample else 1.0),
                        repetition_penalty=float(args.repetition_penalty),
                    )
                    outs = llm.model.generate(batch_prompts, sp, use_tqdm=False)
                    responses = []
                    for out in outs:
                        texts = [o.text for o in out.outputs]
                        if len(texts) < args.num_samples:
                            texts = texts + [""] * (args.num_samples - len(texts))
                        responses.append(texts[: args.num_samples])
                else:
                    responses = []
                    for gp in batch_prompts:
                        inputs = tokenizer(gp, return_tensors="pt").to(hf_model.device)
                        with torch.inference_mode():
                            output_ids = hf_model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=args.do_sample,
                                temperature=args.temperature if args.do_sample else None,
                                top_p=args.top_p if args.do_sample else None,
                                pad_token_id=tokenizer.eos_token_id,
                                repetition_penalty=float(args.repetition_penalty),
                                num_return_sequences=int(args.num_samples),
                            )
                        decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output_ids]
                        responses.append(decoded)

                for row_data, resp_texts in zip(batch_rows, responses):
                    if pbar is not None:
                        pbar.update(1)
                    answer_target = _answer_target_text(row_data["answer"])
                    logp_val = None
                    if args.use_vllm:
                        logp_val = _compute_answer_logp_vllm_sync(
                            vllm_model=llm.model,
                            tokenizer=tokenizer,
                            prompt_text=row_data["meta"]["gen_prompt"],
                            answer_target=answer_target,
                            prompt_logprobs_k=args.prompt_logprobs_k,
                        )
                    else:
                        logp_val = _compute_answer_logp_transformers(
                            model=hf_model,
                            tokenizer=tokenizer,
                            prompt_text=row_data["meta"]["gen_prompt"],
                            answer_target=answer_target,
                        )
                    _process_one(row_data, resp_texts, logp_val, answer_target)

        gen_f.write("\n]\n")
        eval_f.write("\n]\n")

    if pbar is not None:
        pbar.close()

    acc_at_k = (correct_at_k / total) if total else 0.0
    member_acc_at_k = (member_correct_at_k / member_total) if member_total else None
    nonmember_acc_at_k = (nonmember_correct_at_k / nonmember_total) if nonmember_total else None
    member_logp_mean = float(np.nanmean(member_logp)) if member_logp else None
    nonmember_logp_mean = float(np.nanmean(nonmember_logp)) if nonmember_logp else None

    summary = {
        "model": args.model,
        "dataset_path": args.dataset_path,
        "processed": int(total),
        "k": int(args.num_samples),
        "correct_at_k": int(correct_at_k),
        "acc_at_k": round(float(acc_at_k), 6),
        "member": {
            "total": int(member_total),
            "correct_at_k": int(member_correct_at_k),
            "acc_at_k": None if member_acc_at_k is None else round(float(member_acc_at_k), 6),
            "mean_logp_answer_given_prompt": _none_if_nan(member_logp_mean),
        },
        "nonmember": {
            "total": int(nonmember_total),
            "correct_at_k": int(nonmember_correct_at_k),
            "acc_at_k": None if nonmember_acc_at_k is None else round(float(nonmember_acc_at_k), 6),
            "mean_logp_answer_given_prompt": _none_if_nan(nonmember_logp_mean),
        },
        "gen_output": args.gen_output,
        "eval_output": args.eval_output,
    }

    summary_path = Path(args.eval_output).with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote generations to {args.gen_output}")
    print(f"Wrote evaluations to {args.eval_output}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

