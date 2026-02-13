#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.paths import REPORT_DIR
from pipeline.plots import generate_all_plots
from pipeline.q1_metrics import extract_q1_metrics_from_notebook
from pipeline.q2_metrics import load_q2_metrics
from pipeline.report_macros import write_generated_metrics_tex

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def execute_notebook_inplace(path: Path) -> None:
    run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(path),
        ],
        cwd=path.parent,
    )


def smoke(compile_report: bool = False) -> None:
    q1 = extract_q1_metrics_from_notebook(write=True)
    q2 = load_q2_metrics(write=True)
    plot_paths = generate_all_plots(q1, q2)
    macros_path = write_generated_metrics_tex(q1, q2)

    print("Generated plots:")
    for key, value in plot_paths.items():
        print(f"  - {key}: {value}")
    print(f"Generated macros: {macros_path}")

    if compile_report:
        compile_report_pdf()


def compile_report_pdf() -> None:
    run(["latexmk", "-xelatex", "main.tex"], cwd=REPORT_DIR)


def full_q1(compile_report: bool = False) -> None:
    execute_notebook_inplace(ROOT / "nlp-hw4-q1.ipynb")
    smoke(compile_report=compile_report)


def full_q2(compile_report: bool = False) -> None:
    execute_notebook_inplace(ROOT / "NLP_HW4_Q2.ipynb")
    q1 = extract_q1_metrics_from_notebook(write=True)
    q2 = load_q2_metrics(write=True)
    if q2.get("source") != "lm_eval_json":
        raise RuntimeError(
            "full-q2 requires real lm_eval JSON results, but fallback source was used."
        )
    generate_all_plots(q1, q2)
    write_generated_metrics_tex(q1, q2)
    if compile_report:
        compile_report_pdf()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CA4 reproducible artifact and report pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        default="smoke",
        choices=["smoke", "all-smoke", "full-q1", "full-q2", "report"],
        help="Pipeline command",
    )
    parser.add_argument(
        "--compile-report",
        action="store_true",
        help="Compile report PDF after generating artifacts and plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "smoke":
        smoke(compile_report=args.compile_report)
    elif args.command == "all-smoke":
        smoke(compile_report=True)
    elif args.command == "full-q1":
        full_q1(compile_report=args.compile_report)
    elif args.command == "full-q2":
        full_q2(compile_report=args.compile_report)
    elif args.command == "report":
        compile_report_pdf()
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
