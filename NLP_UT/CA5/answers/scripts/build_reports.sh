#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="$ROOT_DIR/answers/reports"
BUILD_DIR="$REPORT_DIR/build"

mkdir -p "$BUILD_DIR"

pushd "$REPORT_DIR" >/dev/null
xelatex -interaction=nonstopmode -halt-on-error -output-directory "$BUILD_DIR" "Q1.tex"
xelatex -interaction=nonstopmode -halt-on-error -output-directory "$BUILD_DIR" "Q1.tex"
xelatex -interaction=nonstopmode -halt-on-error -output-directory "$BUILD_DIR" "Q2.tex"
xelatex -interaction=nonstopmode -halt-on-error -output-directory "$BUILD_DIR" "Q2.tex"
popd >/dev/null

cp -f "$BUILD_DIR/Q1.pdf" "$REPORT_DIR/Q1.pdf"
cp -f "$BUILD_DIR/Q2.pdf" "$REPORT_DIR/Q2.pdf"

echo "Built reports: $REPORT_DIR/Q1.pdf and $REPORT_DIR/Q2.pdf"
