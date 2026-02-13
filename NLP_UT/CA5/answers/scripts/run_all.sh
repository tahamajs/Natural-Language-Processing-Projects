#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /Users/tahamajs/Documents/uni/venv/bin/activate

python "$ROOT_DIR/answers/scripts/preflight.py" --strict-live
python -m pip install --no-cache-dir -r "$ROOT_DIR/answers/requirements.txt"

python "$ROOT_DIR/answers/scripts/run_q1.py"
python "$ROOT_DIR/answers/scripts/run_q2.py"

bash "$ROOT_DIR/answers/scripts/build_reports.sh"

cd "$ROOT_DIR/answers"
rm -f NLP_HW5.zip
zip -r NLP_HW5.zip Q1 Q2 reports scripts \
  -x "*.DS_Store" "__MACOSX/*" \
  "*/.env" "*/.keys.json" "*/__pycache__/*" "*.pyc"

echo "Build complete: $ROOT_DIR/answers/NLP_HW5.zip"
