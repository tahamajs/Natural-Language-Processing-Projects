from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage

from q2_agent import build_agent


def _load_scenarios(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Scenario input file not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values: list[str] = []
            for item in payload:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict):
                    values.append(str(item.get("query", "")))
            return [value for value in values if value.strip()]
        raise ValueError("Scenario JSON must be a list.")
    if suffix == ".csv":
        df = pd.read_csv(source)
        if "query" not in df.columns:
            raise ValueError("Scenario CSV must include a 'query' column.")
        return [str(value).strip() for value in df["query"].fillna("").tolist() if str(value).strip()]
    if suffix in {".txt", ".md"}:
        lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
        return [line for line in lines if line]
    raise ValueError(f"Unsupported scenario file extension: {suffix}")


def _extract_tool_names(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = str(call.get("name", "")).strip()
                if name:
                    names.append(name)
        msg_type = getattr(message, "type", "")
        if msg_type == "tool":
            tool_name = str(getattr(message, "name", "")).strip()
            if tool_name:
                names.append(tool_name)
    unique = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _extract_final_answer(messages: list[Any]) -> str:
    for message in reversed(messages):
        msg_type = getattr(message, "type", "")
        if msg_type == "ai":
            return str(getattr(message, "content", "")).strip()
    return ""


def run_scenarios(scenarios_path: str, output_csv_path: str) -> str:
    scenarios = _load_scenarios(scenarios_path)
    if not scenarios:
        raise RuntimeError("No scenarios found in input.")

    config_path = os.getenv("Q2_CONFIG_PATH", "answers/Q2/config.json")
    agent = build_agent(config_path)

    rows: list[dict[str, Any]] = []
    for idx, query in enumerate(scenarios, start=1):
        started = time.time()
        state = agent.invoke({"messages": [HumanMessage(content=query)]})
        elapsed = time.time() - started

        messages = state.get("messages", []) if isinstance(state, dict) else []
        tool_names = _extract_tool_names(messages)
        answer = _extract_final_answer(messages)

        rows.append(
            {
                "scenario_id": idx,
                "query": query,
                "answer": answer,
                "tool_names": ",".join(tool_names),
                "tool_calls_count": len(tool_names),
                "latency_seconds": elapsed,
            }
        )

    output = Path(output_csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    return str(output)
