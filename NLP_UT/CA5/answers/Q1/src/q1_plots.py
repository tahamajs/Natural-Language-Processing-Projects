from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NODE_COLUMNS = [
    "rewrite",
    "classify_intent",
    "extract_metadata",
    "context_retrieve",
    "rerank",
    "generate_answer",
]


def _save_current(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)


def _plot_timing_summary(df: pd.DataFrame, output_dir: Path) -> list[str]:
    results: list[str] = []
    available = [col for col in NODE_COLUMNS if col in df.columns]
    if not available:
        return results

    avg = df[available].mean(numeric_only=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=avg.index, y=avg.values, palette="Blues_r")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Average seconds")
    plt.title("Average Node Timing")
    for idx, value in enumerate(avg.values):
        plt.text(idx, value + 1e-4, f"{value:.4f}", ha="center", va="bottom")
    results.append(_save_current(output_dir / "avg_node_timings.png"))

    melted = df.melt(value_vars=available, var_name="node", value_name="seconds")
    plt.figure(figsize=(10, 5))
    sns.violinplot(x="node", y="seconds", data=melted, inner="quart")
    plt.xticks(rotation=30, ha="right")
    plt.title("Timing Distribution per Node")
    results.append(_save_current(output_dir / "node_timing_distributions.png"))

    if "total" in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df["total"].dropna(), bins=20, kde=True)
        plt.title("Total Latency Distribution")
        plt.xlabel("Seconds")
        results.append(_save_current(output_dir / "total_latency_histogram.png"))

    return results


def _load_eval_df(metrics_path: Path) -> pd.DataFrame | None:
    candidate = metrics_path.with_name("rag_evaluation_report.xlsx")
    if candidate.exists():
        return pd.read_excel(candidate)
    fallback = metrics_path.with_name("rag_evaluation_report.csv")
    if fallback.exists():
        return pd.read_csv(fallback)
    return None


def _plot_eval_summary(df_eval: pd.DataFrame, output_dir: Path) -> list[str]:
    results: list[str] = []
    metric_columns = [col for col in ["faithfulness", "answer_relevancy"] if col in df_eval.columns]
    if not metric_columns:
        return results

    plt.figure(figsize=(8, 4))
    means = df_eval[metric_columns].mean(numeric_only=True)
    sns.barplot(x=means.index, y=means.values)
    plt.ylim(0, 1)
    plt.ylabel("Mean score")
    plt.title("RAGAS Metric Means")
    for idx, value in enumerate(means.values):
        plt.text(idx, value + 0.02, f"{value:.3f}", ha="center")
    results.append(_save_current(output_dir / "rag_eval_metric_means.png"))

    melted = df_eval.melt(value_vars=metric_columns, var_name="metric", value_name="score")
    plt.figure(figsize=(8, 4))
    sns.violinplot(x="metric", y="score", data=melted, inner="quart")
    sns.stripplot(x="metric", y="score", data=melted, color="black", alpha=0.35, jitter=True)
    plt.ylim(0, 1)
    plt.title("RAGAS Metric Distribution")
    results.append(_save_current(output_dir / "rag_eval_metric_distribution.png"))

    if all(col in df_eval.columns for col in ["faithfulness", "answer_relevancy"]):
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=df_eval, x="faithfulness", y="answer_relevancy", alpha=0.7)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Faithfulness vs Answer Relevancy")
        results.append(_save_current(output_dir / "rag_eval_scatter.png"))

    return results


def generate_all(input_metrics_path: str, output_dir: str) -> list[str]:
    sns.set_theme(style="whitegrid")
    metrics_path = Path(input_metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")
    output_path = Path(output_dir)

    timing_df = pd.read_csv(metrics_path)
    produced = _plot_timing_summary(timing_df, output_path)

    eval_df = _load_eval_df(metrics_path)
    if eval_df is not None and not eval_df.empty:
        produced.extend(_plot_eval_summary(eval_df, output_path))

    if not produced:
        raise RuntimeError("No plots were generated.")
    return produced
