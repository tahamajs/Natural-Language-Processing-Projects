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

cd "${ROOT_DIR}"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m jupyter nbconvert \
  --to notebook \
  --execute answer/HW3.ipynb \
  --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=-1

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
