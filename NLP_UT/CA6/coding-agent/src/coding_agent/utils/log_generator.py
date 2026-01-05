"""Helpers to generate simulated agent logs and aggregate them for reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from coding_agent.agent.session import InteractiveSession


def generate_for_project(project_path: Path) -> Dict[str, Any]:
    sess = InteractiveSession(str(project_path))
    entry = None
    # Use asyncio.run to call the async helper
    import asyncio

    entry = asyncio.run(sess.simulate_agent_run("Fix the failing tests"))

    return entry


def generate_all(root: Path) -> List[Dict[str, Any]]:
    results = []
    test_projects_dir = root / "test_projects"
    for proj in test_projects_dir.iterdir():
        if proj.is_dir():
            try:
                entry = generate_for_project(proj)
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
