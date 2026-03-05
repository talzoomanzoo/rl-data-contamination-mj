import os


#!/usr/bin/env python3
"""
Standalone script to evaluate existing generations in a JSON file.
This script can evaluate responses that were already generated without
needing to regenerate them.
"""

import json
import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def extract_answer_from_response(response_text):
    """Extract the answer from a generated response.
    
    Looks for answers in \boxed{} format, or tries to find the final answer
    in the response text.
    
    Args:
        response_text: The generated response text
        
    Returns:
        The extracted answer as a string, or None if not found
    """
    if not response_text:
        return None
    
    # First, try to extract from \boxed{} format
    # Pattern: \boxed{answer} or \\boxed{answer} (escaped backslash)
    patterns = [
        r'\\boxed\{([^}]+)\}',  # Escaped backslash (in JSON strings)
        r'boxed\{([^}]+)\}',    # Without backslash
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            # Take the last match (usually the final answer)
            answer = matches[-1].strip()
            # Remove any LaTeX formatting
            answer = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', answer)
            answer = re.sub(r'\$([^$]+)\$', r'\1', answer)
            return answer.strip()
    
    # If no boxed answer found, try to find the last number or common answer patterns
    # Look for patterns like "The answer is X" or "Final answer: X"
    answer_patterns = [
        r'(?:answer|Answer|ANSWER)[\s:]+([^\n\.]+)',
        r'(?:final|Final|FINAL)[\s]+(?:answer|Answer|ANSWER)[\s:]+([^\n\.]+)',
        r'is[\s]+([0-9]+)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        if matches:
            answer = matches[-1].strip()
            # Clean up the answer
            answer = re.sub(r'[^\d\w\s]', '', answer)
            if answer:
                return answer.strip()
    
    return None

def normalize_answer(answer):
    """Normalize an answer for comparison.
    
    Removes whitespace, converts to lowercase, and removes common formatting.
    
    Args:
        answer: The answer string to normalize
        
    Returns:
        Normalized answer string
    """
    if answer is None:
        return None
    
    # Convert to string and strip
    answer = str(answer).strip()
    
    # Remove LaTeX formatting
    answer = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\$([^$]+)\$', r'\1', answer)
    
    # Remove common punctuation and whitespace
    answer = re.sub(r'[^\d\w]', '', answer)
    answer = answer.lower()
    
    return answer

def evaluate_response(generated_response, ground_truth_answer):
    """Evaluate whether a generated response is correct.
    
    Args:
        generated_response: The generated response text
        ground_truth_answer: The correct answer from the dataset
        
    Returns:
        Dictionary with evaluation results:
        - is_correct: Boolean indicating if the answer is correct
        - extracted_answer: The answer extracted from the response
        - ground_truth: The ground truth answer
        - normalized_match: Whether normalized answers match
    """
    if not ground_truth_answer:
        return {
            "is_correct": False,
            "extracted_answer": None,
            "ground_truth": None,
            "normalized_match": False,
            "error": "No ground truth answer provided"
        }
    
    extracted_answer = extract_answer_from_response(generated_response)
    
    if extracted_answer is None:
        return {
            "is_correct": False,
            "extracted_answer": None,
            "ground_truth": str(ground_truth_answer),
            "normalized_match": False,
            "error": "Could not extract answer from response"
        }
    
    # Normalize both answers for comparison
    normalized_extracted = normalize_answer(extracted_answer)
    normalized_ground_truth = normalize_answer(ground_truth_answer)
    
    is_correct = normalized_extracted == normalized_ground_truth
    
    return {
        "is_correct": is_correct,
        "extracted_answer": extracted_answer,
        "ground_truth": str(ground_truth_answer),
        "normalized_match": is_correct
    }

def find_model_keys(question_item):
    """Find all model response keys in a question item.
    
    Excludes evaluation keys, error keys, and standard question fields.
    
    Args:
        question_item: A dictionary representing a question
        
    Returns:
        List of model keys found
    """
    standard_fields = {'Question', 'question', 'answer', 'Answer', 'problem', 'prompt', 'text', 'query'}
    model_keys = []
    
    for key in question_item.keys():
        # Skip standard fields
        if key in standard_fields:
            continue
        # Skip evaluation keys
        if key.endswith('_eval'):
            continue
        # Skip error keys
        if key.endswith('_error'):
            continue
        # If it's a string value (likely a response), consider it a model key
        if isinstance(question_item[key], str) and len(question_item[key]) > 50:
            model_keys.append(key)
    
    return model_keys

def _sanitize_filename(value):
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', value).strip('_')

def _select_column(row, candidates, arg_value=None):
    if arg_value:
        return row.get(arg_value)
    for name in candidates:
        if name in row and row[name] is not None:
            return row[name]
    return None

def _select_nested_answer(row, candidates, nested_key):
    if not nested_key:
        return None
    nested = row.get(nested_key)
    if isinstance(nested, dict):
        for name in candidates:
            if name in nested and nested[name] is not None:
                return nested[name]
    return None

def _build_prompt(tokenizer, question_text):
    user_prompt = (
        "Please answer the following math question. "
        "You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
        f"Question:\n{question_text}\n\n"
    )
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt

def generate_responses(args):
    df = pd.read_parquet(args.dataset_path)
    model_key = args.model
    answers_map = None
    if args.answers_json:
        if not os.path.exists(args.answers_json):
            print(f"WARN: answers_json not found: {args.answers_json}. Skipping.")
        else:
            with open(args.answers_json, "r") as f:
                answers_map = json.load(f)

    # --- Compatibility patch for older Transformers ---
    # Some tokenizer configs (e.g. Qwen) store `extra_special_tokens` as a list.
    # Transformers<=4.55 can crash expecting a dict (special_tokens.keys()).
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        _orig_set_model_specific_special_tokens = PreTrainedTokenizerBase._set_model_specific_special_tokens

        def _set_model_specific_special_tokens_compat(self, special_tokens=None):
            if isinstance(special_tokens, list):
                return
            return _orig_set_model_specific_special_tokens(self, special_tokens=special_tokens)

        PreTrainedTokenizerBase._set_model_specific_special_tokens = _set_model_specific_special_tokens_compat
    except Exception:
        pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except AttributeError as e:
        # Fallback for the same "extra_special_tokens list" issue.
        if "keys" in str(e):
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=True, extra_special_tokens={}
            )
        else:
            raise
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    generations = []
    question_candidates = ["Question", "question", "problem", "prompt", "text", "query"]
    answer_candidates = ["answer", "Answer", "solution", "final_answer", "target"]

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating ({args.model})"):
        question_text = _select_column(row, question_candidates, args.question_col)
        answer_text = _select_column(row, answer_candidates, args.answer_col)
        if answer_text is None:
            answer_text = _select_nested_answer(row, answer_candidates, args.answer_nested_col)
        if answer_text is None and answers_map is not None:
            lookup_key = None
            extra_info = row.get("extra_info")
            if isinstance(extra_info, dict) and "index" in extra_info:
                lookup_key = str(int(extra_info["index"]))
            elif "index" in row:
                lookup_key = str(int(row["index"]))
            else:
                lookup_key = str(row_idx)
            answer_text = answers_map.get(lookup_key)
        if question_text is None:
            continue

        prompt = _build_prompt(tokenizer, str(question_text))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        generations.append({
            "Question": str(question_text),
            "answer": None if answer_text is None else str(answer_text),
            model_key: generated_text,
        })

    return generations, model_key

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate existing generations in a JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Hugging Face model name or path."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="Path to eval dataset (.parquet)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Input JSON file containing questions with generated responses"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save the results (default: overwrites input)"
    )
    parser.add_argument(
        "--gen_output",
        type=str,
        default=None,
        help="Output JSON file to save generated responses."
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Output JSON file to save evaluation results."
    )
    parser.add_argument(
        "--question_col",
        type=str,
        default=None,
        help="Optional column name for questions in the parquet dataset."
    )
    parser.add_argument(
        "--answer_col",
        type=str,
        default=None,
        help="Optional column name for answers in the parquet dataset."
    )
    parser.add_argument(
        "--answer_nested_col",
        type=str,
        default=None,
        help="Optional nested column (dict) to look for answer keys."
    )
    parser.add_argument(
        "--answers_json",
        type=str,
        default=None,
        help="Optional JSON map of index->answer for missing labels."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling (default: greedy decoding)."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=None,
        help="Specific model keys to evaluate (default: auto-detect all model keys)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index of questions to process (default: 0)"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Ending index of questions to process (default: all)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation results"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for generation."
    )
    
    args = parser.parse_args()

    if not args.input:
        if not args.model or not args.dataset_path:
            raise SystemExit("Provide --input OR both --model and --dataset_path.")
        dataset_tag = _sanitize_filename(Path(args.dataset_path).stem)
        model_tag = _sanitize_filename(args.model)
        base_dir = Path(args.eval_output or args.gen_output or ".")
        base_dir = base_dir.parent if base_dir.suffix else base_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        if args.gen_output is None:
            args.gen_output = str(base_dir / f"{dataset_tag}__{model_tag}__generations.json")
        if args.eval_output is None:
            args.eval_output = str(base_dir / f"{dataset_tag}__{model_tag}__evaluated.json")

        print(f"\nGenerating responses using model: {args.model}")
        print(f"Dataset: {args.dataset_path}")
        generations, model_key = generate_responses(args)

        print(f"\nSaving generations to: {args.gen_output}")
        with open(args.gen_output, "w") as f:
            json.dump(generations, f, ensure_ascii=False, indent=4)

        args.input = args.gen_output
        args.output = args.eval_output
    else:
        if args.output is None:
            args.output = args.input
    
    # Load questions
    print(f"\nLoading questions from: {args.input}")
    try:
        with open(args.input, 'r') as f:
            questions_data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(questions_data, dict):
            questions_data = [questions_data]
        elif not isinstance(questions_data, list):
            raise ValueError(f"Unexpected data format in {args.input}")
        
        print(f"Found {len(questions_data)} questions in file")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Determine which questions to process
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(questions_data)
    questions_to_process = questions_data[start_idx:end_idx]
    
    print(f"\nProcessing questions {start_idx} to {end_idx-1} ({len(questions_to_process)} questions)")

    has_any_answer = any(
        (q.get("answer") or q.get("Answer")) for q in questions_to_process
    )
    if not has_any_answer:
        print("\nNo ground-truth answers found; skipping evaluation.")
        output_file = args.output if args.output else args.input
        print(f"Saving generations to: {output_file}")
        with open(output_file, "w") as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=4)
        return
    
    # Determine which models to evaluate
    if args.models:
        models_to_evaluate = args.models
        print(f"Evaluating models: {', '.join(models_to_evaluate)}")
    else:
        # Auto-detect models from the first question
        if questions_to_process:
            models_to_evaluate = find_model_keys(questions_to_process[0])
            print(f"Auto-detected models: {', '.join(models_to_evaluate)}")
        else:
            print("No questions to process")
            return
    
    print("=" * 80)
    
    # Process each question
    for idx, question_item in enumerate(questions_to_process):
        global_idx = start_idx + idx
        print(f"\n[Question {global_idx + 1}/{len(questions_data)}]")
        
        question_text = question_item.get('Question') or question_item.get('question', '')
        print(f"Question: {question_text[:100]}..." if len(question_text) > 100 else f"Question: {question_text}")
        
        ground_truth = question_item.get('answer') or question_item.get('Answer')
        if not ground_truth:
            print(f"  ⚠ No ground truth answer found, skipping evaluation")
            continue
        
        # Evaluate each model
        for model_key in models_to_evaluate:
            eval_key = f"{model_key}_eval"
            
            # Check if evaluation already exists
            if not args.overwrite and eval_key in question_item:
                existing_eval = question_item[eval_key]
                if existing_eval and existing_eval.get("is_correct") is not None:
                    print(f"  → Skipping {model_key} (evaluation already exists)")
                    continue
            
            # Check if response exists
            if model_key not in question_item:
                print(f"  → Skipping {model_key} (no response found)")
                continue
            
            response = question_item[model_key]
            if not response or not isinstance(response, str):
                print(f"  → Skipping {model_key} (invalid response)")
                continue
            
            print(f"  → Evaluating {model_key}...")
            
            eval_result = evaluate_response(response, ground_truth)
            question_item[eval_key] = eval_result
            
            status = "✓ CORRECT" if eval_result["is_correct"] else "✗ INCORRECT"
            print(f"  {status}")
            print(f"    Extracted: {eval_result['extracted_answer']}")
            print(f"    Expected:  {eval_result['ground_truth']}")
            if not eval_result["is_correct"] and "error" in eval_result:
                print(f"    Error: {eval_result['error']}")
    
    # Save results
    output_file = args.output if args.output else args.input
    
    print(f"\n{'=' * 80}")
    print(f"\nSaving results to: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Results saved successfully!")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  Total questions: {len(questions_data)}")
    print(f"  Questions processed: {len(questions_to_process)}")
    
    # Evaluation summary
    print(f"\n📈 Evaluation Summary:")
    summary = {
        "total_questions": len(questions_data),
        "questions_processed": len(questions_to_process),
        "models": {},
    }
    for model_key in models_to_evaluate:
        eval_key = f"{model_key}_eval"
        
        correct_count = 0
        total_count = 0
        no_answer_count = 0
        
        for question_item in questions_to_process:
            if eval_key in question_item:
                eval_result = question_item[eval_key]
                total_count += 1
                if eval_result.get("is_correct", False):
                    correct_count += 1
                elif eval_result.get("extracted_answer") is None:
                    no_answer_count += 1
        
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            print(f"  {model_key}:")
            print(f"    Correct: {correct_count}/{total_count} ({accuracy:.1f}%)")
            print(f"    No answer extracted: {no_answer_count}")
            summary["models"][model_key] = {
                "correct": correct_count,
                "total": total_count,
                "accuracy_percent": round(accuracy, 3),
                "no_answer_extracted": no_answer_count,
            }
        else:
            print(f"  {model_key}: No evaluations performed")
            summary["models"][model_key] = {
                "correct": 0,
                "total": 0,
                "accuracy_percent": None,
                "no_answer_extracted": no_answer_count,
            }

    summary_path = Path(output_file).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
