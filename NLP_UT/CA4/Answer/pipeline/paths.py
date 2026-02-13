from __future__ import annotations

from pathlib import Path

ANSWER_DIR = Path(__file__).resolve().parents[1]
CA4_DIR = ANSWER_DIR.parent
PIPELINE_DIR = ANSWER_DIR / "pipeline"
SCRIPTS_DIR = ANSWER_DIR / "scripts"
ARTIFACTS_DIR = ANSWER_DIR / "artifacts"
REPORT_DIR = ANSWER_DIR / "report"
REPORT_FIGURES_DIR = REPORT_DIR / "figures"

Q1_NOTEBOOK_PATH = ANSWER_DIR / "nlp-hw4-q1.ipynb"
Q2_NOTEBOOK_PATH = ANSWER_DIR / "NLP_HW4_Q2.ipynb"

Q1_ARTIFACT_PATH = ARTIFACTS_DIR / "q1_metrics.json"
Q2_ARTIFACT_PATH = ARTIFACTS_DIR / "q2_metrics.json"

GENERATED_METRICS_TEX_PATH = REPORT_DIR / "generated_metrics.tex"

Q1_EM_FIGURE_PATH = REPORT_FIGURES_DIR / "q1_em_comparison.png"
Q1_ERROR_FIGURE_PATH = REPORT_FIGURES_DIR / "q1_error_types.png"
Q2_IFEVAL_FIGURE_PATH = REPORT_FIGURES_DIR / "q2_ifeval_comparison.png"

Q2_LM_EVAL_CANDIDATES = [
    ANSWER_DIR / "eval_results_base" / "results.json",
    ANSWER_DIR / "eval_results_finetuned" / "results.json",
    ANSWER_DIR / "eval_results" / "base" / "results.json",
    ANSWER_DIR / "eval_results" / "finetuned" / "results.json",
    ANSWER_DIR / "results_base.json",
    ANSWER_DIR / "results_finetuned.json",
]


def ensure_output_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
