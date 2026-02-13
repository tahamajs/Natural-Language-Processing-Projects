from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

# Matplotlib needs a writable config dir in this environment.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import (
    Q1_EM_FIGURE_PATH,
    Q1_ERROR_FIGURE_PATH,
    Q2_IFEVAL_FIGURE_PATH,
    ensure_output_dirs,
)


Q2_LABELS = {
    "prompt_level_strict_acc": "Prompt Strict",
    "inst_level_strict_acc": "Inst Strict",
    "prompt_level_loose_acc": "Prompt Loose",
    "inst_level_loose_acc": "Inst Loose",
}


def _m(model_payload: dict[str, Any], key: str) -> float:
    return float(model_payload[key])


def generate_q1_em_plot(q1_payload: dict[str, Any], out_path: Path = Q1_EM_FIGURE_PATH) -> Path:
    ensure_output_dirs()
    bart = q1_payload["models"]["BART"]
    gpt2 = q1_payload["models"]["GPT-2"]

    labels = ["BART Dev", "BART Test", "GPT-2 Dev", "GPT-2 Test"]
    raw_vals = [
        _m(bart, "dev_raw_em"),
        _m(bart, "test_raw_em"),
        _m(gpt2, "dev_raw_em"),
        _m(gpt2, "test_raw_em"),
    ]
    norm_vals = [
        _m(bart, "dev_norm_em"),
        _m(bart, "test_norm_em"),
        _m(gpt2, "dev_norm_em"),
        _m(gpt2, "test_norm_em"),
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_raw = ax.bar(x - width / 2, raw_vals, width, label="Raw EM", color="#2E86AB")
    bars_norm = ax.bar(x + width / 2, norm_vals, width, label="Normalized EM", color="#F18F01")

    ax.set_title("Q1 Exact Match Comparison")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars in (bars_raw, bars_norm):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_q1_error_plot(q1_payload: dict[str, Any], out_path: Path = Q1_ERROR_FIGURE_PATH) -> Path:
    ensure_output_dirs()
    bart_errors = q1_payload["errors"]["BART"]
    gpt2_errors = q1_payload["errors"]["GPT-2"]

    keys = list(bart_errors.keys())
    bart_vals = [int(bart_errors[k]) for k in keys]
    gpt2_vals = [int(gpt2_errors[k]) for k in keys]

    x = np.arange(len(keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, bart_vals, width, label="BART", color="#5DADE2")
    ax.bar(x + width / 2, gpt2_vals, width, label="GPT-2", color="#EC7063")

    ax.set_title("Q1 Error Type Counts (Dev)")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=40, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_q2_ifeval_plot(q2_payload: dict[str, Any], out_path: Path = Q2_IFEVAL_FIGURE_PATH) -> Path:
    ensure_output_dirs()
    metrics = q2_payload["metrics"]

    ordered_keys = [
        "prompt_level_strict_acc",
        "inst_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_loose_acc",
    ]
    labels = [Q2_LABELS[k] for k in ordered_keys]
    base_vals = [float(metrics[k]["base"]) * 100.0 for k in ordered_keys]
    fine_vals = [float(metrics[k]["fine_tuned"]) * 100.0 for k in ordered_keys]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, base_vals, width, label="Base", color="#7F8C8D")
    ax.bar(x + width / 2, fine_vals, width, label="Fine-tuned", color="#27AE60")

    ax.set_title("Q2 IFEval Comparison")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(base_vals + fine_vals + [1.0]) + 10)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_q1_plots_from_metrics(
    q1_payload: dict[str, Any],
    em_path: Path = Q1_EM_FIGURE_PATH,
    error_path: Path = Q1_ERROR_FIGURE_PATH,
) -> Dict[str, str]:
    em = generate_q1_em_plot(q1_payload, em_path)
    err = generate_q1_error_plot(q1_payload, error_path)
    return {"q1_em": str(em), "q1_error": str(err)}


def generate_all_plots(
    q1_payload: dict[str, Any],
    q2_payload: dict[str, Any],
) -> Dict[str, str]:
    q1_paths = generate_q1_plots_from_metrics(q1_payload)
    q2 = generate_q2_ifeval_plot(q2_payload)
    return {
        **q1_paths,
        "q2_ifeval": str(q2),
    }
