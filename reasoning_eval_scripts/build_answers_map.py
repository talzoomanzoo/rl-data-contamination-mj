#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def _write_map(answer_map, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(answer_map, f, ensure_ascii=False, indent=2)


def _map_from_dataframe(df, answer_col, key_col=None):
    if answer_col not in df.columns:
        raise SystemExit(f"answer_col '{answer_col}' not found in dataset columns.")
    if key_col and key_col not in df.columns:
        raise SystemExit(f"key_col '{key_col}' not found in dataset columns.")

    answer_map = {}
    for row_idx, row in df.iterrows():
        key = row_idx if key_col is None else row[key_col]
        answer = row[answer_col]
        if pd.isna(answer):
            continue
        answer_map[str(key)] = str(answer)
    return answer_map


def _map_from_json(data, answer_col, key_col=None):
    if isinstance(data, dict):
        # Already in map form.
        return {str(k): str(v) for k, v in data.items()}
    if not isinstance(data, list):
        raise SystemExit("JSON input must be a list or dict.")
    if answer_col is None:
        raise SystemExit("answer_col is required for JSON list input.")

    answer_map = {}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        key = idx if key_col is None else item.get(key_col)
        if key is None:
            continue
        answer = item.get(answer_col)
        if answer is None:
            continue
        answer_map[str(key)] = str(answer)
    return answer_map


def main():
    parser = argparse.ArgumentParser(
        description="Build index->answer JSON map for evaluation."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to answer source (csv/parquet/json)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON map path."
    )
    parser.add_argument(
        "--answer_col",
        type=str,
        default=None,
        help="Column/key containing answers."
    )
    parser.add_argument(
        "--key_col",
        type=str,
        default=None,
        help="Optional column/key to use as map key. Defaults to row index."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    suffix = input_path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        df = pd.read_csv(input_path, sep="\t" if suffix == ".tsv" else ",")
        answer_map = _map_from_dataframe(df, args.answer_col, args.key_col)
    elif suffix == ".parquet":
        df = pd.read_parquet(input_path)
        answer_map = _map_from_dataframe(df, args.answer_col, args.key_col)
    elif suffix == ".json":
        with open(input_path, "r") as f:
            data = json.load(f)
        answer_map = _map_from_json(data, args.answer_col, args.key_col)
    else:
        raise SystemExit(f"Unsupported input format: {suffix}")

    if not answer_map:
        raise SystemExit("No answers found; check answer_col/key_col.")

    _write_map(answer_map, output_path)
    print(f"Wrote {len(answer_map)} answers to {output_path}")


if __name__ == "__main__":
    main()
