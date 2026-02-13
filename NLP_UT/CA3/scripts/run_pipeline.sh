#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="/Users/tahamajs/Documents/uni/venv"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Virtual environment not found at ${VENV_PATH}" >&2
  exit 1
fi

source "${VENV_PATH}/bin/activate"
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p "${MPLCONFIGDIR}"
export CA3_RUNTIME_PROFILE="${CA3_RUNTIME_PROFILE:-full}"

cd "${ROOT_DIR}"

echo "Using CA3 runtime profile: ${CA3_RUNTIME_PROFILE}"

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

python scripts/execute_notebook.py answer/HW3.ipynb --kernel python3

for required_plot in report/top_words.png report/wordcloud.png; do
  if [[ ! -s "${required_plot}" ]]; then
    echo "Missing or empty plot file: ${required_plot}" >&2
    exit 1
  fi
done

cd "${ROOT_DIR}/report"
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex

echo "Pipeline completed successfully."
