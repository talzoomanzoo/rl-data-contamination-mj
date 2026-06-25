#!/usr/bin/env python3
"""Add all-layers LaRA-robust RSM/RSI/DC entries to an existing evaluation_summary.json.

Reads per-layer scores from rep_stiff_scores.json in the same directory (produced by
evaluate_all_methods.py), computes rep_stiff_lara_robust_{rsm,rsi,dc}, and merges metrics
into evaluation_summary.json without re-running RepStiff on generated_data.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluate_all_methods import (
    _compute_rep_stiff_derived_columns,
    _ordered_rep_stiff_layers,
    evaluate_performance,
    evaluate_performance_pop,
)


PATCH_METHODS = (
    "rep_stiff_lara_robust",
    "rep_stiff_lara_robust_rsm",
    "rep_stiff_lara_robust_rsi",
    "rep_stiff_lara_robust_dc",
)


def _layers_from_rep_stiff_scores(df: pd.DataFrame) -> list[str]:
    layer_names = {c.split("_", 1)[1] for c in df.columns if c.startswith("rsm_")}
    return _ordered_rep_stiff_layers(list(layer_names))


def _dataframe_from_rep_stiff_scores(rep_stiff_path: str) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.DataFrame(json.loads(open(rep_stiff_path, encoding="utf-8").read()))
    layers = _layers_from_rep_stiff_scores(raw)
    df = raw.copy()
    for layer in layers:
        df[f"rep_stiff_rsm_{layer}_score"] = raw[f"rsm_{layer}"]
        df[f"rep_stiff_directional_collapse_{layer}_score"] = raw[f"directional_collapse_{layer}"]
        df[f"rep_stiff_rsi_{layer}_score"] = raw[f"rsi_{layer}"]
    return df, layers


def _evaluate_method(df_scores: pd.DataFrame, method_name: str) -> dict:
    df_method = df_scores.dropna(subset=[method_name])
    y_true = df_method["ground_truth_label"].values
    y_scores = df_method[method_name].values

    overall_perf = evaluate_performance_pop(y_true, y_scores)

    breakdown = {}
    mean_auc = 0.0
    for source, group in df_method.groupby("data_source"):
        group_perf = evaluate_performance(
            group["ground_truth_label"].values,
            group[method_name].values,
        )
        breakdown[source] = group_perf
        if not np.isnan(group_perf["roc_auc"]):
            mean_auc += group_perf["roc_auc"]
    mean_auc /= len(breakdown) if breakdown else 1

    return {
        "overall_performance": overall_perf,
        "mean_auc": mean_auc,
        "breakdown_by_source": breakdown,
    }


def patch_run_dir(run_dir: str, *, rewrite_rep_stiff_scores: bool = True) -> None:
    summary_path = os.path.join(run_dir, "evaluation_summary.json")
    rep_stiff_path = os.path.join(run_dir, "rep_stiff_scores.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(summary_path)
    if not os.path.isfile(rep_stiff_path):
        raise FileNotFoundError(rep_stiff_path)

    df_scores, layers = _dataframe_from_rep_stiff_scores(rep_stiff_path)
    ordered_layers = _ordered_rep_stiff_layers(layers)
    args = SimpleNamespace(
        rep_stiff_lara_robust_layer_window="all",
        rep_stiff_lara_robust_dc_weight=1.0,
        rep_stiff_lara_eps=1e-8,
        rep_stiff_lara_clean_ref=None,
        rep_stiff_lara_mix_beta=0.65,
        rep_stiff_combined_weights=None,
        rep_stiff_combined_fixed=False,
        rep_stiff_combined_v4_alpha=0.0,
        rep_stiff_v6_mix_beta=0.65,
    )

    rep_stiff_scores_list = df_scores.to_dict(orient="records") if rewrite_rep_stiff_scores else None
    _compute_rep_stiff_derived_columns(df_scores, args, ordered_layers, rep_stiff_scores_list)

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    for method_name in PATCH_METHODS:
        summary[method_name] = _evaluate_method(df_scores, method_name)
        perf = summary[method_name]["overall_performance"]
        print(
            f"  {method_name}: auc={perf['roc_auc']:.4f} "
            f"tpr@5%={perf['tpr_at_fpr_5']:.4f}"
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    if rewrite_rep_stiff_scores and rep_stiff_scores_list is not None:
        with open(rep_stiff_path, "w", encoding="utf-8") as f:
            json.dump(rep_stiff_scores_list, f, indent=2)

    print(f"Patched {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Directories containing evaluation_summary.json and rep_stiff_scores.json",
    )
    parser.add_argument(
        "--no-rewrite-rep-stiff-scores",
        action="store_true",
        help="Only update evaluation_summary.json",
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        print(f"\n--> {run_dir}")
        patch_run_dir(run_dir, rewrite_rep_stiff_scores=not args.no_rewrite_rep_stiff_scores)


if __name__ == "__main__":
    main()
