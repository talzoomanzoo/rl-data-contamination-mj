from dotenv import load_dotenv
import os
import json
import argparse
import requests
from pathlib import Path

load_dotenv()

# Get OpenRouter API key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")

def get_available_datasets():
    """Get list of available math datasets from the eval folder."""
    eval_dir = Path(__file__).parent / "eval" / "chat_benchmarks"
    datasets = []
    
    for item in eval_dir.iterdir():
        if item.is_dir() and (item / "data").exists():
            datasets.append(item.name)
    
    return sorted(datasets)

def load_dataset(dataset_name):
    """Load dataset from /data/dataset_name/test.json or train.json."""
    data_dir = Path(f"../data/{dataset_name}")
    
    if not data_dir.exists():
        raise ValueError(f"Dataset directory '{data_dir}' not found")
    
    questions = []
    
    # Try test.json first, then train.json
    file_path = data_dir / "test.json"
    if not file_path.exists():
        file_path = data_dir / "train.json"
    
    if not file_path.exists():
        raise ValueError(f"Neither test.json nor train.json found in {data_dir}")
    
    # Load the file (handle both JSON and JSONL formats)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # First, try to parse as regular JSON (could be pretty-printed)
        try:
            data = json.loads(content)
            if isinstance(data, list):
                questions = data
            else:
                questions = [data]
        except json.JSONDecodeError:
            # If JSON parsing fails, try JSONL format (one JSON object per line)
            questions = []
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
    
    return questions

def extract_question_text(question_data):
    """Extract the question text from various dataset formats."""
    # Try different field names that might contain the question
    possible_fields = ['problem', 'question', 'prompt', 'text', 'query']
    
    for field in possible_fields:
        if field in question_data:
            return question_data[field]
    
    # If none found, return the first string value
    for value in question_data.values():
        if isinstance(value, str) and len(value) > 20:
            return value
    
    return str(question_data)

def generate_similar_questions(original_question, num_questions=5, model="openai/gpt-4o-mini"):
    """Generate similar questions using OpenRouter API."""
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")
    
    prompt = f"""You are a math problem generator. Given an original math problem, create {num_questions} similar problems that:

1. Follow the EXACT same structure and solution method as the original
2. Use DIFFERENT numerical values (change all numbers to make the problem unique)
3. Maintain the same difficulty level
4. Have the same type of solution approach
5. Are valid, solvable problems

Original Problem:
{original_question}

Generate {num_questions} similar problems. For each problem:
- Change ALL numerical values to create unique scenarios
- Keep the problem structure and mathematical concepts identical
- Ensure the problem remains solvable and realistic
- Make sure the new numbers create valid mathematical relationships

Output ONLY a JSON array of {num_questions} similar problems, where each element is a string containing the full problem text. Do not include solutions or explanations, only the problems.

Format your response as:
["Problem 1 text here...", "Problem 2 text here...", ...]
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Similar Question Generator"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.8,
        "max_tokens": 4000
    }
    
    print("\nCalling OpenRouter API with gpt-4o-mini...")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    
    # Parse the JSON array from the response
    try:
        # Try to find JSON array in the response
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            similar_questions = json.loads(json_str)
        else:
            # If no JSON array found, split by newlines and clean up
            similar_questions = [q.strip() for q in content.split('\n') if q.strip() and len(q.strip()) > 20]
    except json.JSONDecodeError:
        # Fallback: split by newlines and clean up
        similar_questions = [q.strip() for q in content.split('\n') if q.strip() and len(q.strip()) > 20]
    
    return similar_questions

def main():
    parser = argparse.ArgumentParser(
        description="Generate similar math questions with different numerical setups using OpenRouter API"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to use (e.g., AIME24, MATH500, AMC23)"
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=1,
        help="End index for the slice of questions to process (e.g., --question-id 1 processes questions 0 to 1, default: 1)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of similar questions to generate (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="similar_questions.json",
        help="Output file to save the generated questions (optional, will print to console if not specified)"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/o4-mini",
        help="Model to use for generating similar questions (default: openai/gpt-4o-mini)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="Output format: json (single file) or jsonl (one question per line)"
    )
    
    args = parser.parse_args()
    
    # List available datasets if requested
    if args.list_datasets:
        print("\nAvailable datasets:")
        for dataset in get_available_datasets():
            print(f"  - {dataset}")
        return
    
    # Validate dataset argument
    if not args.dataset:
        print("Error: --dataset argument is required")
        print("\nUse --list-datasets to see available datasets")
        print("\nAvailable datasets:")
        for dataset in get_available_datasets():
            print(f"  - {dataset}")
        return
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    try:
        questions = load_dataset(args.dataset)
        print(f"Found {len(questions)} questions in dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Validate question ID (end index for slice)
    if args.question_id < 1:
        print(f"Error: question-id must be at least 1 (got {args.question_id})")
        return
    if args.question_id > len(questions):
        print(f"Error: question-id {args.question_id} is out of range (dataset has {len(questions)} questions, max end index: {len(questions)})")
        return
    
    # Get slice of questions to process (0 to question_id)
    questions_to_process = questions[0:args.question_id]
    print(f"\nProcessing questions 0 to {args.question_id-1} ({len(questions_to_process)} question(s))")
    
    all_output_data = []
    
    # Process each question in the slice
    for question_idx, question_data in enumerate(questions_to_process):
        original_question = extract_question_text(question_data)
        
        print(f"\n{'='*80}")
        print(f"Original Question (ID: {question_idx}):")
        print("-" * 80)
        print(original_question)
        print("-" * 80)
        
        # Generate similar questions
        try:
            similar_questions = generate_similar_questions(original_question, args.num_questions, args.model)
            
            print(f"\n✓ Generated {len(similar_questions)} similar questions:")
            print("=" * 80)
            
            output_data = {
                "dataset": args.dataset,
                "original_question_id": question_idx,
                "original_question": original_question,
                "similar_questions": []
            }
            
            for i, question in enumerate(similar_questions, 1):
                print(f"\n[Similar Question {i}]")
                print(question)
                print()
                
                output_data["similar_questions"].append({
                    "id": i,
                    "question": question
                })
            
            all_output_data.append(output_data)
            print(f"\n✓ Successfully processed question {question_idx}")
            
        except Exception as e:
            print(f"\n✗ Error generating questions for question {question_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save to file if output path specified
    if args.output and all_output_data:
        if args.format == "jsonl":
            # Save as JSONL (one question per line)
            with open(args.output, 'w') as f:
                for output_data in all_output_data:
                    question_idx = output_data["original_question_id"]
                    original_question = output_data["original_question"]
                    
                    # Write original question
                    f.write(json.dumps({
                        "id": f"{args.dataset}_original_{question_idx}",
                        "question": original_question,
                        "source": args.dataset,
                        "type": "original"
                    }) + '\n')
                    
                    # Write similar questions
                    for similar_q in output_data["similar_questions"]:
                        f.write(json.dumps({
                            "id": f"{args.dataset}_similar_{question_idx}_{similar_q['id']}",
                            "question": similar_q["question"],
                            "source": args.dataset,
                            "type": "similar",
                            "original_id": question_idx
                        }) + '\n')
            print(f"\n✓ Results saved to: {args.output} (JSONL format)")
        else:
            # Save as JSON (default)
            # If only one question, save in original format; otherwise save as array
            if len(all_output_data) == 1:
                with open(args.output, 'w') as f:
                    json.dump(all_output_data[0], f, indent=2)
            else:
                with open(args.output, 'w') as f:
                    json.dump(all_output_data, f, indent=2)
            print(f"\n✓ Results saved to: {args.output} (JSON format)")
        
        print("=" * 80)
        print(f"\n✓ Successfully processed {len(all_output_data)} question(s)!")

if __name__ == "__main__":
    main()
