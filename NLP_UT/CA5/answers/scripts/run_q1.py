from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import gdown

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
Q1_SRC = PROJECT_ROOT / "answers" / "Q1" / "src"
if str(Q1_SRC) not in sys.path:
    sys.path.append(str(Q1_SRC))

from q1_eval import run_benchmark, run_ragas_evaluation
from q1_pipeline import build_index
from q1_plots import generate_all


GOOGLE_DRIVE_FILE_ID = "13hpV5jRwRNYSTzAt90bsMR0lYEIWfyxB"


def _ensure_dataset(data_dir: Path, zip_path: Path) -> Path:
    if data_dir.exists() and any(data_dir.glob("*.pdf")):
        return data_dir
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, str(zip_path), quiet=False)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)

    nested_pdfs = list(data_dir.rglob("*.pdf"))
    if not nested_pdfs:
        raise RuntimeError("No PDF files were found after extracting Q1 dataset.")

    for pdf in nested_pdfs:
        if pdf.parent != data_dir:
            target = data_dir / pdf.name
            if not target.exists():
                pdf.replace(target)

    if not any(data_dir.glob("*.pdf")):
        raise RuntimeError("Dataset extraction did not produce top-level PDFs.")
    return data_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="answers/Q1/CA5_data")
    parser.add_argument("--dataset-zip", default="answers/Q1/CA5_data.zip")
    parser.add_argument("--index-path", default="answers/Q1/data/indexed_dataset.pkl")
    parser.add_argument("--benchmark-input", default="answers/Q1/data/benchmark_queries.json")
    parser.add_argument("--timings-csv", default="answers/Q1/data/pipeline_timings.csv")
    parser.add_argument("--eval-xlsx", default="answers/Q1/data/rag_evaluation_report.xlsx")
    parser.add_argument("--plots-dir", default="answers/reports/figures/q1")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_zip = Path(args.dataset_zip)
    _ensure_dataset(dataset_dir, dataset_zip)

    index_path = build_index(str(dataset_dir), args.index_path)
    print(f"Q1 index written: {index_path}")

    benchmark_input = Path(args.benchmark_input)
    benchmark_source = str(benchmark_input) if benchmark_input.exists() else None
    timings_csv = run_benchmark(benchmark_source, args.timings_csv, index_path=index_path)
    print(f"Q1 benchmark written: {timings_csv}")

    evaluation_xlsx = run_ragas_evaluation(timings_csv, args.eval_xlsx)
    print(f"Q1 evaluation written: {evaluation_xlsx}")

    plot_paths = generate_all(timings_csv, args.plots_dir)
    for path in plot_paths:
        print(f"Q1 plot written: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
