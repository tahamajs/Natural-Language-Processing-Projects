from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
Q2_SRC = PROJECT_ROOT / "answers" / "Q2" / "src"
if str(Q2_SRC) not in sys.path:
    sys.path.append(str(Q2_SRC))

from q2_plots import generate_all
from q2_scenarios import run_scenarios


DEFAULT_SCENARIOS = [
    "Find a flight from Tehran to Dubai for next Friday.",
    "Find a hotel in Paris for 3 nights starting tomorrow.",
    "What is the weather in Barcelona next Monday?",
    "Convert currency from Iran to UAE today.",
    "Plan a 3-day trip to Istanbul focused on history and food.",
]


def _ensure_scenarios(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"query": query} for query in DEFAULT_SCENARIOS]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="answers/Q2/results/scenarios.json")
    parser.add_argument("--output-csv", default="answers/Q2/results/scenario_results.csv")
    parser.add_argument("--plots-dir", default="answers/reports/figures/q2")
    args = parser.parse_args()

    scenarios_path = _ensure_scenarios(Path(args.scenarios))
    csv_path = run_scenarios(str(scenarios_path), args.output_csv)
    print(f"Q2 scenarios written: {csv_path}")

    plots = generate_all(csv_path, args.plots_dir)
    for plot in plots:
        print(f"Q2 plot written: {plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
