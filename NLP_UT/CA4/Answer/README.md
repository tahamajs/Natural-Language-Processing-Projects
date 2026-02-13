# CA4 Reproducible Pipeline

This folder includes a smoke-first pipeline to generate report metrics artifacts, plot files, and LaTeX metric macros from the current project state.

## Structure

- `pipeline/`: shared code used by scripts and notebooks
- `scripts/run_pipeline.py`: CLI entrypoint
- `artifacts/q1_metrics.json`: normalized Q1 metrics payload
- `artifacts/q2_metrics.json`: normalized Q2 metrics payload
- `report/figures/*.png`: generated report plots
- `report/generated_metrics.tex`: generated LaTeX macros consumed by report

## Setup

From `/Users/tahamajs/Documents/uni/NLP/nlp-assignments-spring-2023/NLP_UT/CA4/Answer`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-smoke.txt
```

For full training/eval runs:

```bash
pip install -r requirements-full.txt
```

## Commands

Run smoke pipeline (default command):

```bash
python scripts/run_pipeline.py
```

Equivalent explicit smoke command:

```bash
python scripts/run_pipeline.py smoke
```

Smoke + report compile:

```bash
python scripts/run_pipeline.py all-smoke
```

Compile report only:

```bash
python scripts/run_pipeline.py report
```

Optional full notebook execution:

```bash
python scripts/run_pipeline.py full-q1 --compile-report
python scripts/run_pipeline.py full-q2 --compile-report
```

## Q2 Source Behavior

`q2_metrics.json` includes a `source` field:

- `lm_eval_json`: real `lm_eval` result JSON files were found
- `report_table_fallback`: fallback parsed from `report/Q2.tex` table values

Current default remains fallback unless real eval files exist.

## Expected smoke outputs

After `python scripts/run_pipeline.py smoke`:

- `/Users/tahamajs/Documents/uni/NLP/nlp-assignments-spring-2023/NLP_UT/CA4/Answer/artifacts/q1_metrics.json`
- `/Users/tahamajs/Documents/uni/NLP/nlp-assignments-spring-2023/NLP_UT/CA4/Answer/artifacts/q2_metrics.json`
- `/Users/tahamajs/Documents/uni/NLP/nlp-assignments-spring-2023/NLP_UT/CA4/Answer/report/figures/q1_em_comparison.png`
- `/Users/tahamajs/Documents/uni/NLP/nlp-assignments-spring-2023/NLP_UT/CA4/Answer/report/figures/q1_error_types.png`
- `/Users/tahamajs/Documents/uni/NLP/nlp-assignments-spring-2023/NLP_UT/CA4/Answer/report/figures/q2_ifeval_comparison.png`
- `/Users/tahamajs/Documents/uni/NLP/nlp-assignments-spring-2023/NLP_UT/CA4/Answer/report/generated_metrics.tex`
