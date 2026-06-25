#!/usr/bin/env python3
"""Compute metric-ablation AUC/TPR@5% and print LaTeX table rows (Init epoch)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluate_all_methods import (
    _compute_rep_stiff_derived_columns,
    _ordered_rep_stiff_layers,
    evaluate_performance_pop,
)

LARA_METRICS = [
    ("rep_stiff_rsm", "rsm", "signed"),
    ("rep_stiff_directional_collapse", "directional_collapse", "absolute"),
    ("rep_stiff_rsi", "rsi", "negated"),
]

# (use_rsm, use_dc, use_rsi) checkmarks in table order
ABLATIONS = [
    (False, False, True),
    (False, True, False),
    (True, False, False),
    (False, True, True),
    (True, False, True),
    (True, True, False),
    (True, True, True),
]

RUNS = {
    "Eurus": {
        "Init": Path(
            "final_results/eurus-epoch0-step8/eurus_member/_all__all_samples/blank_k_sweep/k1"
        ),
        "E1": Path(
            "final_results/eurus-epoch1-step15/eurus_member/_all__all_samples/blank_k_sweep/k1"
        ),
        "E2": Path(
            "final_results/Eurus-2-7B-PRIME_eurus/eurus_member/_all__all_samples/blank_k_sweep/k1"
        ),
    },
    "LIMR": {
        "Init": Path("final_results/GAIR__LIMR/limr/_all__all_samples/blank_k_sweep/k1"),
    },
    "OLMO": {
        "Init": Path(
            "final_results/Olmo-3.1-7B-RL-Zero-Math_olmoe/olmoe_member/_all__all_samples/blank_k_sweep/k1"
        ),
    },
}

EPOCHS = ["Init", "E1", "E2"]
MODELS = ["Eurus", "LIMR", "OLMO"]


def _load_df(run_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.DataFrame(json.loads((run_dir / "rep_stiff_scores.json").read_text()))
    layers = sorted(
        {c.split("_", 1)[1] for c in raw.columns if c.startswith("rsm_")},
        key=lambda name: int(name[1:]),
    )
    df = raw.copy()
    for layer in layers:
        df[f"rep_stiff_rsm_{layer}_score"] = raw[f"rsm_{layer}"]
        df[f"rep_stiff_directional_collapse_{layer}_score"] = raw[f"directional_collapse_{layer}"]
        df[f"rep_stiff_rsi_{layer}_score"] = raw[f"rsi_{layer}"]
    return df, _ordered_rep_stiff_layers(layers)


def _compute_lara_subset(df: pd.DataFrame, layers: list[str], use_rsm: bool, use_dc: bool, use_rsi: bool) -> np.ndarray:
    """Replicate evaluate_all_methods LaRA-robust all-layers for a metric subset."""
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
    _compute_rep_stiff_derived_columns(df, args, layers, None)

    if use_rsm and use_dc and use_rsi:
        return pd.to_numeric(df["rep_stiff_lara_robust"], errors="coerce").to_numpy()
    if use_rsm and not use_dc and not use_rsi:
        return pd.to_numeric(df["rep_stiff_lara_robust_rsm"], errors="coerce").to_numpy()
    if use_dc and not use_rsm and not use_rsi:
        return pd.to_numeric(df["rep_stiff_lara_robust_dc"], errors="coerce").to_numpy()
    if use_rsi and not use_rsm and not use_dc:
        return pd.to_numeric(df["rep_stiff_lara_robust_rsi"], errors="coerce").to_numpy()

    # Pairwise: compute on the fly via duplicated _compute_lara logic in derived columns
    # Extend df with temporary column by re-running subset of metrics only.
    from evaluate_all_methods import _compute_rep_stiff_derived_columns as _  # noqa: F401

    # Import inner function by re-executing derived columns with custom active metrics
    # Use evaluate_all_methods module-level pattern: patch via second pass.
    active = []
    if use_rsm:
        active.append(LARA_METRICS[0])
    if use_dc:
        active.append(LARA_METRICS[1])
    if use_rsi:
        active.append(LARA_METRICS[2])

    # Inline minimal LaRA (same as patch script / evaluate_all_methods).
    label_arr = pd.to_numeric(df["ground_truth_label"], errors="coerce").to_numpy()
    clean_mask = label_arr == 0
    if clean_mask.sum() < 2:
        clean_mask = np.ones(len(df), dtype=bool)
    n_rows = len(df)
    log_metrics = {"rsm", "directional_collapse", "rsi"}
    eps = 1e-8
    M, L = len(active), len(layers)
    zhat_stack = np.full((n_rows, M, L), np.nan, dtype=float)

    for li, layer_name in enumerate(layers):
        for mi, (col_prefix, metric_short, alignment) in enumerate(active):
            col = f"{col_prefix}_{layer_name}_score"
            raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            raw_stats = np.sign(raw) * np.log1p(np.abs(raw)) if metric_short in log_metrics else raw
            clean_vals = raw_stats[clean_mask]
            clean_vals = clean_vals[np.isfinite(clean_vals)]
            if clean_vals.size < 2:
                continue
            med = float(np.median(clean_vals))
            mad = float(np.median(np.abs(clean_vals - med)))
            sigma = 1.4826 * mad
            z = (raw_stats - med) / (sigma + eps)
            if alignment == "absolute":
                zhat = np.abs(z)
            elif alignment == "negated":
                zhat = -z
            else:
                zhat = z
            zhat_stack[:, mi, li] = zhat

    with np.errstate(invalid="ignore"):
        return np.nanmean(np.abs(zhat_stack).reshape(n_rows, -1), axis=1)


def _fmt(auc: float, tpr: float) -> str:
    if not (np.isfinite(auc) and np.isfinite(tpr)):
        return " &  "
    return f" & {auc:.2f} & {tpr:.2f}"


def main() -> None:
    results: dict[tuple[str, str, bool, bool, bool], tuple[float, float]] = {}

    for model in MODELS:
        for epoch, run_dir in RUNS[model].items():
            if not (run_dir / "rep_stiff_scores.json").is_file():
                continue
            df, layers = _load_df(run_dir)
            y = pd.to_numeric(df["ground_truth_label"], errors="coerce").to_numpy()
            for use_rsm, use_dc, use_rsi in ABLATIONS:
                scores = _compute_lara_subset(df.copy(), layers, use_rsm, use_dc, use_rsi)
                perf = evaluate_performance_pop(y, scores)
                results[(model, epoch, use_rsm, use_dc, use_rsi)] = (
                    float(perf["roc_auc"]),
                    float(perf["tpr_at_fpr_5"]),
                )
                print(
                    f"{model} {epoch} RSM={use_rsm} DC={use_dc} RSI={use_rsi}: "
                    f"auc={perf['roc_auc']:.4f} tpr={perf['tpr_at_fpr_5']:.4f}"
                )

    print("\n% --- LaTeX body (paste into table) ---")
    for use_rsm, use_dc, use_rsi in ABLATIONS:
        rsm_tex = r"{\color{green!60!black}\ding{51}}" if use_rsm else r"{\color{red}\ding{55}}"
        dc_tex = r"{\color{green!60!black}\ding{51}}" if use_dc else r"{\color{red}\ding{55}}"
        rsi_tex = r"{\color{green!60!black}\ding{51}}" if use_rsi else r"{\color{red}\ding{55}}"
        row = f"{rsm_tex} & {dc_tex} & {rsi_tex}"
        for model in MODELS:
            for epoch in EPOCHS:
                key = (model, epoch, use_rsm, use_dc, use_rsi)
                if key in results:
                    auc, tpr = results[key]
                    row += _fmt(auc, tpr)
                else:
                    row += " &  & "
        if use_rsm and use_dc and use_rsi:
            row = r"\rowcolor{lightblue}" + "\n" + row
        print(row + r" \\")


if __name__ == "__main__":
    main()
