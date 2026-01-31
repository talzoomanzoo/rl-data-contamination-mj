from dotenv import load_dotenv
import os
import json
import argparse
import requests
import re
from pathlib import Path

load_dotenv()

# Get OpenRouter API key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")

def load_incomplete_questions(input_file):
    """Load incomplete questions from a JSON or JSONL file."""
    questions = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # First, try to parse as regular JSON
        try:
            data = json.loads(content)
            if isinstance(data, list):
                questions = data
            else:
                questions = [data]
        except json.JSONDecodeError:
            # If JSON parsing fails, try JSONL format
            questions = []
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    return questions

def extract_incomplete_question(question_data):
    """Extract the incomplete question text from various formats."""
    # Priority: incomplete_question > question > other fields
    if isinstance(question_data, str):
        return question_data
    
    if isinstance(question_data, dict):
        # Check for incomplete_question field first
        if 'incomplete_question' in question_data:
            return question_data['incomplete_question']
        
        # Then check other common fields
        possible_fields = ['question', 'problem', 'prompt', 'text', 'query']
        for field in possible_fields:
            if field in question_data:
                return question_data[field]
        
        # If none found, return the first string value
        for value in question_data.values():
            if isinstance(value, str) and len(value) > 20:
                return value
    
    return str(question_data)

def reorder_clauses(text):
    """Reorder clauses in a sentence while preserving [BLANK] position."""
    # Split by common clause separators (periods, semicolons, commas before conjunctions)
    # This is a simple heuristic - for more complex cases, we'll use LLM
    sentences = re.split(r'([.!?]\s+)', text)
    
    if len(sentences) <= 2:
        return text
    
    # Reorder sentences (skip the last empty string if present)
    non_empty = [s for s in sentences if s.strip()]
    if len(non_empty) <= 1:
        return text
    
    # Simple reordering: swap adjacent pairs
    reordered = non_empty.copy()
    for i in range(0, len(reordered) - 1, 2):
        if i + 1 < len(reordered):
            reordered[i], reordered[i + 1] = reordered[i + 1], reordered[i]
    
    return ''.join(reordered)

def remove_redundant_phrasing(text):
    """Remove redundant phrasing while preserving [BLANK] position."""
    # Simple patterns for redundant phrasing
    patterns = [
        (r'\s+and\s+also\s+', ' and '),
        (r'\s+,\s*,\s*', ', '),
        (r'\s+which\s+is\s+also\s+', ' which '),
        (r'\s+that\s+is\s+also\s+', ' that '),
    ]
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result

def apply_formatting_changes(text):
    """Apply minor formatting changes while preserving [BLANK] position."""
    # Minor formatting changes
    changes = [
        (r'\s+', ' '),  # Normalize whitespace
        (r'\.\s+\.', '.'),  # Remove double periods
        (r',\s*,', ','),  # Remove double commas
    ]
    
    result = text
    for pattern, replacement in changes:
        result = re.sub(pattern, replacement, result)
    
    return result.strip()

def generate_paraphrase_variant(incomplete_question, model="openai/gpt-4o-mini"):
    """Generate a paraphrased variant using LLM while keeping [BLANK] position."""
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")
    
    prompt = f"""You are a text rewriter that creates paraphrased versions of math problems.

Given an incomplete math problem with [BLANK] placeholders, create a paraphrased version that:

1. Preserves the EXACT position and meaning of [BLANK] - do NOT move or change [BLANK]
2. Uses different wording and phrasing while maintaining the same mathematical meaning
3. Keeps the same structure and logical flow
4. Does NOT reveal what the blank should be
5. Maintains all mathematical relationships and constraints

Incomplete Problem:
{incomplete_question}

Output ONLY the paraphrased version with [BLANK] in the same position. Do not include explanations or notes.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Paraphrase Generator"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    result = response.json()
    paraphrased = result['choices'][0]['message']['content'].strip()
    
    # Clean up the response (remove quotes if present)
    if paraphrased.startswith('"') and paraphrased.endswith('"'):
        paraphrased = paraphrased[1:-1]
    if paraphrased.startswith("'") and paraphrased.endswith("'"):
        paraphrased = paraphrased[1:-1]
    
    return paraphrased


def generate_incomplete_variants_batch(
    incomplete_question, num_variants=5, model="openai/gpt-4o-mini"
):
    """
    Generate N paraphrased variants in a single API call.
    Returns list of variant strings (length <= num_variants).
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")

    if "[BLANK]" not in incomplete_question and "[blank]" not in incomplete_question.lower():
        raise ValueError("Input question must contain [BLANK] placeholder")

    prompt = f"""You are a text rewriter that creates paraphrased versions of math problems.

Given an incomplete math problem with [BLANK] placeholders, create {num_variants} paraphrased versions that:
1. Preserve the EXACT position and meaning of [BLANK] - do NOT move or change [BLANK]
2. Use different wording and phrasing while maintaining the same mathematical meaning
3. Keep the same structure and logical flow
4. Do NOT reveal what the blank should be
5. Maintain all mathematical relationships and constraints

Incomplete Problem:
{incomplete_question}

Output ONLY a JSON array of {num_variants} strings with the paraphrased versions. Do not include explanations or notes.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Paraphrase Generator (Batch)"
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    result = response.json()
    content = result['choices'][0]['message']['content']

    try:
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            variants = json.loads(json_str)
        else:
            raise json.JSONDecodeError("No JSON array found", content, 0)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON array from batch response: {e}")

    if not isinstance(variants, list):
        raise ValueError("Batch response is not a JSON array")

    return variants[:num_variants]

def generate_incomplete_variants(incomplete_question, num_variants=5, model="openai/gpt-4o-mini"):
    """Generate M semantics-preserving perturbations of an incomplete question.
    
    Uses multiple strategies:
    1. Clause reordering
    2. Paraphrasing (LLM) while keeping [BLANK] position
    3. Minor formatting changes
    4. Remove redundant phrasing
    """
    
    if "[BLANK]" not in incomplete_question and "[blank]" not in incomplete_question.lower():
        raise ValueError("Input question must contain [BLANK] placeholder")
    
    variants = []
    strategies = []
    
    # Strategy 1: Paraphrase using LLM (generate multiple)
    num_llm_variants = max(2, num_variants // 2)  # Use LLM for at least half
    
    for i in range(num_llm_variants):
        try:
            variant = generate_paraphrase_variant(incomplete_question, model)
            if variant and variant != incomplete_question:
                variants.append(variant)
                strategies.append("paraphrase_llm")
        except Exception as e:
            print(f"Warning: Failed to generate LLM paraphrase variant {i+1}: {e}")
    
    # Strategy 2: Clause reordering
    try:
        reordered = reorder_clauses(incomplete_question)
        if reordered != incomplete_question and len(variants) < num_variants:
            variants.append(reordered)
            strategies.append("clause_reorder")
    except Exception as e:
        print(f"Warning: Failed to reorder clauses: {e}")
    
    # Strategy 3: Remove redundant phrasing
    try:
        cleaned = remove_redundant_phrasing(incomplete_question)
        if cleaned != incomplete_question and len(variants) < num_variants:
            variants.append(cleaned)
            strategies.append("remove_redundant")
    except Exception as e:
        print(f"Warning: Failed to remove redundant phrasing: {e}")
    
    # Strategy 4: Formatting changes
    try:
        formatted = apply_formatting_changes(incomplete_question)
        if formatted != incomplete_question and len(variants) < num_variants:
            variants.append(formatted)
            strategies.append("formatting")
    except Exception as e:
        print(f"Warning: Failed to apply formatting changes: {e}")
    
    # If we need more variants, generate additional LLM paraphrases
    while len(variants) < num_variants:
        try:
            variant = generate_paraphrase_variant(incomplete_question, model)
            if variant and variant != incomplete_question and variant not in variants:
                variants.append(variant)
                strategies.append("paraphrase_llm")
        except Exception as e:
            print(f"Warning: Failed to generate additional LLM variant: {e}")
            break
    
    # Ensure we have at least the original if no variants were generated
    if not variants:
        variants = [incomplete_question]
        strategies = ["original"]
    
    # Trim to exactly num_variants
    variants = variants[:num_variants]
    strategies = strategies[:num_variants]
    
    return variants, strategies

def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple semantics-preserving variants of incomplete questions with [BLANK] for RSI calculation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON/JSONL file containing incomplete questions (with [BLANK] placeholders)"
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=None,
        help="End index for the slice of questions to process (e.g., --question-id 5 processes questions 0 to 5). If not specified, processes all questions."
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Start index (0-based) for the slice of questions to process (e.g., --start-id 22 starts from question 22). "
             "Default: 0 (start from the first question)."
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=5,
        help="Number of semantics-preserving variants to generate per incomplete question (M, default: 5, minimal)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="incomplete_variants_rsi.json",
        help="Output JSON file to save all generated variants (default: incomplete_variants_rsi.json). "
             "If the file already exists, entries for the same question will be updated."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model to use for generating paraphrased variants (default: openai/gpt-4o-mini)"
    )
    
    args = parser.parse_args()
    
    # Load incomplete questions
    print(f"\nLoading incomplete questions from: {args.input}")
    try:
        questions = load_incomplete_questions(args.input)
        print(f"Found {len(questions)} incomplete question(s)")
    except Exception as e:
        print(f"Error loading input file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Validate start and end indices
    if args.start_id < 0:
        print(f"Error: start-id must be at least 0 (got {args.start_id})")
        return
    if args.start_id >= len(questions):
        print(f"Error: start-id {args.start_id} is out of range (file has {len(questions)} questions, max index: {len(questions)-1})")
        return

    if args.question_id is not None:
        if args.question_id <= args.start_id:
            print(f"Error: question-id (end index) must be greater than start-id (got question-id={args.question_id}, start-id={args.start_id})")
            return
        if args.question_id > len(questions):
            print(f"Error: question-id {args.question_id} is out of range (file has {len(questions)} questions, max end index: {len(questions)})")
            return
        start_idx = args.start_id
        end_idx = args.question_id
    else:
        start_idx = args.start_id
        end_idx = len(questions)

    questions_to_process = questions[start_idx:end_idx]
    print(f"\nProcessing questions {start_idx} to {end_idx-1} ({len(questions_to_process)} question(s))")
    
    all_output_data = []

    # Prepare output JSON state (load existing file once, reuse per-question)
    existing_entries = []
    output_path = Path(args.output) if args.output else None

    if output_path is not None and output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_entries = json.load(f)
                if not isinstance(existing_entries, list):
                    existing_entries = []
        except Exception:
            # If reading/parsing fails, start fresh
            existing_entries = []

    def make_key(entry: dict):
        """
        Build a stable key for a question based on its metadata.
        This lets us update/overwrite entries for the same question across runs.
        """
        return (
            entry.get("dataset"),
            entry.get("original_question_id"),
            entry.get("similar_question_id"),
            entry.get("type"),
            entry.get("incomplete_question_id"),
        )

    # Index existing entries by key
    by_key = {}
    for e in existing_entries:
        by_key[make_key(e)] = e
    
    # Process each incomplete question
    for offset, question_data in enumerate(questions_to_process):
        question_idx = start_idx + offset  # global index within the full dataset
        incomplete_question = extract_incomplete_question(question_data)
        
        # Extract metadata if available
        metadata = {}
        if isinstance(question_data, dict):
            metadata = {k: v for k, v in question_data.items() 
                       if k not in ['incomplete_question', 'question', 'problem', 'prompt', 'text', 'query']}
        
        print(f"\n{'='*80}")
        print(f"Incomplete Question (ID: {question_idx}):")
        print("-" * 80)
        print(incomplete_question[:200] + "..." if len(incomplete_question) > 200 else incomplete_question)
        print("-" * 80)
        
        # Check if question has [BLANK]
        if "[BLANK]" not in incomplete_question and "[blank]" not in incomplete_question.lower():
            print(f"\n⚠ Warning: Question {question_idx} does not contain [BLANK]. Skipping...")
            continue
        
        # Generate variants
        try:
            variants, strategies = generate_incomplete_variants(
                incomplete_question, 
                args.num_variants, 
                args.model
            )
            
            print(f"\n✓ Generated {len(variants)} variant(s):")
            print("=" * 80)
            
            output_data = {
                "incomplete_question_id": question_idx,
                "original_incomplete_question": incomplete_question,
                "variants": []
            }
            
            # Add metadata if available
            if metadata:
                output_data.update(metadata)
            
            for i, (variant, strategy) in enumerate(zip(variants, strategies), 1):
                print(f"\n[Variant {i} - {strategy}]")
                print(variant[:200] + "..." if len(variant) > 200 else variant)
                print()
                
                output_data["variants"].append({
                    "id": i,
                    "variant": variant,
                    "strategy": strategy
                })
            
            all_output_data.append(output_data)

            # Update in-memory index and immediately write out JSON file
            if output_path is not None:
                by_key[make_key(output_data)] = output_data
                final_entries = list(by_key.values())
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(final_entries, f, ensure_ascii=False, indent=2)

                print(f"\n  ↳ Saved/updated JSON entry for question {question_idx} in {args.output}")
            print(f"\n✓ Successfully processed question {question_idx}")
            
        except Exception as e:
            print(f"\n✗ Error generating variants for question {question_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    if args.output:
        total_entries = len(by_key)
        print(f"\n✓ Final JSON saved/updated in: {args.output} (JSON format, {total_entries} question entries total)")
    print("=" * 80)
    print(f"\n✓ Successfully processed {len(all_output_data)} incomplete question(s) this run!")
    print(f"✓ Generated {sum(len(d['variants']) for d in all_output_data)} total variant(s) this run!")

if __name__ == "__main__":
    main()
