from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from .paths import Q1_ARTIFACT_PATH, Q1_NOTEBOOK_PATH, ensure_output_dirs
from .schemas import Q1MetricsPayload, Q1ModelMetrics

_EXPECTED_ERROR_KEYS = [
    "syntax_error",
    "missing_select",
    "missing_from",
    "missing_where",
    "wrong_table",
    "wrong_column",
    "incomplete",
    "extra_content",
    "correct",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_texts(notebook: dict[str, Any]) -> Iterable[str]:
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if text:
                yield text


def _parse_model_scores(texts: Iterable[str]) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {
        "BART": {},
        "GPT-2": {},
    }
    row_pattern = re.compile(
        r"^(BART|GPT-2)\s+(Dev|Test)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)$"
    )

    for text in texts:
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line.strip())
            match = row_pattern.match(line)
            if not match:
                continue
            model, split, raw_em, norm_em = match.groups()
            split_key = split.lower()
            rows[model][f"{split_key}_raw_em"] = float(raw_em)
            rows[model][f"{split_key}_norm_em"] = float(norm_em)

    return rows


def _parse_training_times(texts: Iterable[str]) -> Dict[str, float]:
    times: Dict[str, float] = {}
    pattern = re.compile(r"(BART|GPT-2):\s*([0-9]+(?:\.[0-9]+)?)\s*Seconds")

    for text in texts:
        for match in pattern.finditer(text):
            model, seconds = match.groups()
            times[model] = float(seconds)

    return times


def _parse_error_counts(texts: Iterable[str]) -> Dict[str, Dict[str, int]]:
    errors = {
        "BART": {key: 0 for key in _EXPECTED_ERROR_KEYS},
        "GPT-2": {key: 0 for key in _EXPECTED_ERROR_KEYS},
    }

    active_model: str | None = None
    section_pattern = re.compile(r"\b(BART|GPT-2)\s*\(Dev\)")
    count_pattern = re.compile(r"^([a-z_]+):\s*([0-9]+)")

    for text in texts:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            section_match = section_pattern.search(line)
            if section_match:
                active_model = section_match.group(1)
                continue
            if not active_model:
                continue
            count_match = count_pattern.match(line)
            if not count_match:
                continue
            key, count = count_match.groups()
            if key in errors[active_model]:
                errors[active_model][key] = int(count)

    return errors


def _require_fields(values: Dict[str, Dict[str, float]], times: Dict[str, float]) -> None:
    required_score_keys = {"dev_raw_em", "dev_norm_em", "test_raw_em", "test_norm_em"}
    for model in ("BART", "GPT-2"):
        score_keys = set(values.get(model, {}).keys())
        missing_scores = required_score_keys - score_keys
        if missing_scores:
            raise ValueError(
                f"Could not find {model} score fields in notebook outputs: {sorted(missing_scores)}"
            )
        if model not in times:
            raise ValueError(f"Could not find {model} training time in notebook outputs")


def save_q1_metrics(payload: dict[str, Any], artifact_path: Path = Q1_ARTIFACT_PATH) -> Path:
    ensure_output_dirs()
    artifact_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return artifact_path


def extract_q1_metrics_from_notebook(
    notebook_path: Path = Q1_NOTEBOOK_PATH,
    artifact_path: Path = Q1_ARTIFACT_PATH,
    write: bool = True,
) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text())
    texts = list(_stream_texts(notebook))

    model_scores = _parse_model_scores(texts)
    training_times = _parse_training_times(texts)
    error_counts = _parse_error_counts(texts)
    _require_fields(model_scores, training_times)

    payload = Q1MetricsPayload(
        generated_at=_utc_now_iso(),
        source="q1_notebook_output",
        models={
            "BART": Q1ModelMetrics(
                dev_raw_em=model_scores["BART"]["dev_raw_em"],
                dev_norm_em=model_scores["BART"]["dev_norm_em"],
                test_raw_em=model_scores["BART"]["test_raw_em"],
                test_norm_em=model_scores["BART"]["test_norm_em"],
                training_time_s=training_times["BART"],
            ),
            "GPT-2": Q1ModelMetrics(
                dev_raw_em=model_scores["GPT-2"]["dev_raw_em"],
                dev_norm_em=model_scores["GPT-2"]["dev_norm_em"],
                test_raw_em=model_scores["GPT-2"]["test_raw_em"],
                test_norm_em=model_scores["GPT-2"]["test_norm_em"],
                training_time_s=training_times["GPT-2"],
            ),
        },
        errors=error_counts,
    ).to_dict()

    if write:
        save_q1_metrics(payload, artifact_path)
    return payload


def _coerce_int_map(values: dict[str, Any] | None) -> dict[str, int]:
    if not values:
        return {key: 0 for key in _EXPECTED_ERROR_KEYS}
    result = {key: 0 for key in _EXPECTED_ERROR_KEYS}
    for key in _EXPECTED_ERROR_KEYS:
        if key in values:
            result[key] = int(values[key])
    return result


def export_q1_metrics_from_runtime(
    bart_results: dict[str, Any],
    gpt2_results: dict[str, Any],
    bart_errors: dict[str, Any] | None = None,
    gpt2_errors: dict[str, Any] | None = None,
    artifact_path: Path = Q1_ARTIFACT_PATH,
    write: bool = True,
) -> dict[str, Any]:
    payload = Q1MetricsPayload(
        generated_at=_utc_now_iso(),
        source="q1_runtime_memory",
        models={
            "BART": Q1ModelMetrics(
                dev_raw_em=float(bart_results["dev_raw_em"]),
                dev_norm_em=float(bart_results["dev_norm_em"]),
                test_raw_em=float(bart_results["test_raw_em"]),
                test_norm_em=float(bart_results["test_norm_em"]),
                training_time_s=float(bart_results["training_time"]),
            ),
            "GPT-2": Q1ModelMetrics(
                dev_raw_em=float(gpt2_results["dev_raw_em"]),
                dev_norm_em=float(gpt2_results["dev_norm_em"]),
                test_raw_em=float(gpt2_results["test_raw_em"]),
                test_norm_em=float(gpt2_results["test_norm_em"]),
                training_time_s=float(gpt2_results["training_time"]),
            ),
        },
        errors={
            "BART": _coerce_int_map(bart_errors),
            "GPT-2": _coerce_int_map(gpt2_errors),
        },
    ).to_dict()

    if write:
        save_q1_metrics(payload, artifact_path)
    return payload
