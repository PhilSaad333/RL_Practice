#!/usr/bin/env python3
"""Analyze Fisher-kernel outliers and save diagnostics alongside results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

WORKSPACE_SEQ_PATH = ROOT / "workspace_sequences.json"
EVAL_SEQ_PATH = ROOT / "evaluation_sequences.json"
KERNEL_UPDATE_PATH = ROOT / "kernel_update.npy"
KERNEL_EVAL_PATH = ROOT / "kernel_eval.npy"
INFLUENCE_UPDATE_PATH = ROOT / "influence_update_self.npy"
INFLUENCE_EVAL_PATH = ROOT / "influence_eval.npy"
UPDATE_OUTLIERS_JSON = ROOT / "kernel_update_top_outliers.json"
EVAL_OUTLIERS_JSON = ROOT / "kernel_eval_top_outliers.json"

OUTPUT_SUMMARY = ROOT / "outlier_pair_details.json"
OUTPUT_CORR = ROOT / "workspace_kernel_metric_correlations.json"
OUTPUT_SCATTER = ROOT / "workspace_kernel_scatter.png"
OUTPUT_EVAL_SCATTER = ROOT / "evaluation_kernel_scatter.png"
OUTPUT_WORKSPACE_METRICS = ROOT / "workspace_sequence_metrics.json"
OUTPUT_EVAL_METRICS = ROOT / "evaluation_sequence_metrics.json"


def load_sequences(path: Path) -> Tuple[List[Dict], Dict[str, Dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {entry["sequence_id"]: entry for entry in data}


def derive_metrics(record: Dict) -> Dict[str, float]:
    entropies = record.get("entropy_per_token") or []
    logprobs = record.get("logprob_per_token") or []
    metrics: Dict[str, float] = {
        "reward": record.get("reward"),
        "advantage": record.get("advantage"),
        "total_logprob": record.get("total_logprob"),
        "total_logq": record.get("total_logq"),
        "response_len": len(record.get("response_tokens") or []),
        "prompt_len": len(record.get("prompt_tokens") or []),
    }
    if entropies:
        ent = np.asarray(entropies, dtype=np.float64)
        metrics.update(
            mean_entropy=float(np.mean(ent)),
            max_entropy=float(np.max(ent)),
            min_entropy=float(np.min(ent)),
            std_entropy=float(np.std(ent)),
        )
    if logprobs:
        lp = np.asarray(logprobs, dtype=np.float64)
        metrics.update(
            mean_logprob=float(np.mean(lp)),
            max_logprob=float(np.max(lp)),
            min_logprob=float(np.min(lp)),
            std_logprob=float(np.std(lp)),
        )
    return metrics


def compute_correlations(values: Dict[str, List[float]], target: List[float]) -> Dict[str, float]:
    correlations: Dict[str, float] = {}
    target_arr = np.asarray(target, dtype=np.float64)
    for metric, vec in values.items():
        if metric in {"max_abs_kernel", "sum_abs_kernel", "row_l2"}:
            continue
        arr = np.asarray(vec, dtype=np.float64)
        if arr.size != target_arr.size or np.isnan(arr).any() or np.isnan(target_arr).any():
            continue
        if np.allclose(arr, arr[0]) or np.allclose(target_arr, target_arr[0]):
            correlations[metric] = float("nan")
            continue
        corr = np.corrcoef(arr, target_arr)[0, 1]
        correlations[metric] = float(corr)
    return correlations


def scatter_plot(x: List[float], y: List[float], xlabel: str, ylabel: str, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=14, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def summarize_outliers(outliers_path: Path, row_map: Dict[str, Dict], col_map: Dict[str, Dict]) -> List[Dict]:
    if not outliers_path.exists():
        return []
    entries = json.loads(outliers_path.read_text(encoding="utf-8"))
    summary: List[Dict] = []
    for entry in entries:
        row_record = row_map.get(entry.get("row_sequence_id"))
        col_record = col_map.get(entry.get("col_sequence_id"))
        def mean_entropy(rec: Dict | None) -> float | None:
            if rec and rec.get("entropy_per_token"):
                return float(np.mean(rec["entropy_per_token"]))
            return None
        summary.append(
            {
                "value": entry.get("value"),
                "row_sequence_id": entry.get("row_sequence_id"),
                "col_sequence_id": entry.get("col_sequence_id"),
                "row_prompt": row_record.get("prompt_text") if row_record else None,
                "row_response": row_record.get("response_text") if row_record else None,
                "row_reward": row_record.get("reward") if row_record else None,
                "row_advantage": row_record.get("advantage") if row_record else None,
                "row_total_logprob": row_record.get("total_logprob") if row_record else None,
                "row_mean_entropy": mean_entropy(row_record),
                "col_prompt": col_record.get("prompt_text") if col_record else None,
                "col_response": col_record.get("response_text") if col_record else None,
                "col_reward": col_record.get("reward") if col_record else None,
                "col_advantage": col_record.get("advantage") if col_record else None,
                "col_total_logprob": col_record.get("total_logprob") if col_record else None,
                "col_mean_entropy": mean_entropy(col_record),
            }
        )
    summary.sort(key=lambda item: abs(item["value"] or 0), reverse=True)
    return summary


def main() -> None:
    workspace_order, workspace_map = load_sequences(WORKSPACE_SEQ_PATH)
    eval_order, eval_map = load_sequences(EVAL_SEQ_PATH)

    kernel_update = np.load(KERNEL_UPDATE_PATH)
    kernel_eval = np.load(KERNEL_EVAL_PATH)
    influence_update = np.load(INFLUENCE_UPDATE_PATH)
    influence_eval = np.load(INFLUENCE_EVAL_PATH)

    # Workspace metrics and correlations
    workspace_metrics: Dict[str, Dict] = {}
    metric_vectors: Dict[str, List[float]] = {}
    abs_kernel = np.abs(kernel_update)
    max_abs_kernel = np.max(abs_kernel, axis=1)
    sum_abs_kernel = np.sum(abs_kernel, axis=1)
    row_norm = np.linalg.norm(kernel_update, axis=1)

    for idx, record in enumerate(workspace_order):
        seq_id = record["sequence_id"]
        metrics = derive_metrics(record)
        metrics.update(
            max_abs_kernel=float(max_abs_kernel[idx]),
            sum_abs_kernel=float(sum_abs_kernel[idx]),
            row_l2=float(row_norm[idx]),
            influence=float(influence_update[idx]) if idx < influence_update.size else None,
        )
        workspace_metrics[seq_id] = metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metric_vectors.setdefault(key, []).append(float(value))

    OUTPUT_WORKSPACE_METRICS.write_text(json.dumps(workspace_metrics, indent=2), encoding="utf-8")

    correlations = compute_correlations(metric_vectors, list(max_abs_kernel))
    OUTPUT_CORR.write_text(json.dumps(correlations, indent=2), encoding="utf-8")

    mean_entropy_vec = [workspace_metrics[rec["sequence_id"]].get("mean_entropy", 0.0) for rec in workspace_order]
    scatter_plot(
        mean_entropy_vec,
        list(max_abs_kernel),
        "mean entropy",
        "max |kernel|",
        "Workspace: mean entropy vs max |kernel|",
        OUTPUT_SCATTER,
    )

    # Evaluation metrics
    eval_metrics: Dict[str, Dict] = {}
    for record in eval_order:
        seq_id = record["sequence_id"]
        eval_metrics[seq_id] = derive_metrics(record)
    OUTPUT_EVAL_METRICS.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    if kernel_eval.size:
        eval_max_abs = np.max(np.abs(kernel_eval), axis=1)
        eval_mean_entropy = [eval_metrics[rec["sequence_id"]].get("mean_entropy", 0.0) for rec in eval_order]
        scatter_plot(
            eval_mean_entropy,
            list(eval_max_abs),
            "mean entropy",
            "max |kernel| (eval row)",
            "Evaluation: mean entropy vs max |kernel|",
            OUTPUT_EVAL_SCATTER,
        )

    outlier_summary = {
        "update_kernel": summarize_outliers(UPDATE_OUTLIERS_JSON, workspace_map, workspace_map),
        "eval_kernel": summarize_outliers(EVAL_OUTLIERS_JSON, eval_map, workspace_map),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(outlier_summary, indent=2), encoding="utf-8")

    print(f"Saved analysis artefacts to {ROOT}")


if __name__ == "__main__":
    main()
