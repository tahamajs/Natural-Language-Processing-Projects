"""Helpers to generate simulated agent logs and aggregate them for reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from coding_agent.agent.session import InteractiveSession


def generate_for_project(project_path: Path, full: bool = True) -> Dict[str, Any]:
    sess = InteractiveSession(str(project_path))
    entry = None
    # Use asyncio.run to call the async helper
    import asyncio

    entry = asyncio.run(sess.simulate_agent_run("Fix the failing tests", full=full))

    return entry


def generate_all(root: Path, full: bool = True) -> List[Dict[str, Any]]:
    results = []
    test_projects_dir = root / "test_projects"
    for proj in test_projects_dir.iterdir():
        if proj.is_dir():
            try:
                entry = generate_for_project(proj, full=full)
                results.append({"project": proj.name, "entry": entry})
            except Exception as e:
                results.append({"project": proj.name, "error": str(e)})
    return results


def aggregate_simulated_logs(root: Path, output_path: Path) -> Path:
    """Aggregate all `simulated_session_*.jsonl` into a single JSONL file at output_path."""
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as fout:
        test_projects_dir = root / "test_projects"
        for proj in test_projects_dir.iterdir():
            log_dir = proj / ".coding_agent_logs"
            if not log_dir.exists():
                continue
            for file in sorted(log_dir.glob("simulated_session_*.jsonl")):
                with file.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)

    return outp


def jsonl_to_csv(jsonl_path: Path, csv_out: Path) -> Path:
    """Convert a JSONL of simulated logs into a CSV summary. Each row is one log entry."""
    import csv

    jsonl_path = Path(jsonl_path)
    csv_out = Path(csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("r", encoding="utf-8") as fin, csv_out.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        writer = csv.writer(fout)
        # header
        writer.writerow([
            "timestamp",
            "thread_id",
            "project",
            "user_message",
            "agent_response",
            "usage_total_tokens",
            "usage_cost_est",
            "tool_calls_json",
        ])
        for line in fin:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # determine project from thread by scanning logs in projects dir (best-effort)
            project = obj.get("project") or ""
            usage = obj.get("usage") or {}
            tool_calls = obj.get("tool_calls") or []
            writer.writerow([
                obj.get("timestamp"),
                obj.get("thread_id"),
                project,
                obj.get("user_message"),
                obj.get("agent_response"),
                usage.get("total_tokens", 0),
                usage.get("cost_est", 0.0),
                json.dumps(tool_calls, ensure_ascii=False),
            ])

    return csv_out


def jsonl_to_markdown(jsonl_path: Path, md_out: Path) -> Path:
    """Convert a JSONL of simulated logs into a Markdown report with per-entry sections."""
    jsonl_path = Path(jsonl_path)
    md_out = Path(md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("r", encoding="utf-8") as fin, md_out.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            fout.write(f"## Entry {i} — {obj.get('timestamp', '')}\n\n")
            fout.write(f"**Project:** {obj.get('project', '')}  \n")
            fout.write(f"**Thread:** {obj.get('thread_id', '')}  \n")
            fout.write(f"**User message:** {obj.get('user_message', '')}  \n\n")
            fout.write(f"**Agent response:** {obj.get('agent_response', '')}  \n\n")
            fout.write("**Tool calls:**\n\n")
            for tc in obj.get("tool_calls", []):
                fout.write(f"- **{tc.get('name')}** — args: `{tc.get('args')}`\n\n")
                if "result" in tc and tc.get("result"):
                    fout.write("```\n")
                    fout.write(str(tc.get("result")) + "\n")
                    fout.write("```\n\n")
                if "error" in tc and tc.get("error"):
                    fout.write(f"**Error:** {tc.get('error')}  \n\n")
            fout.write("---\n\n")

    return md_out
