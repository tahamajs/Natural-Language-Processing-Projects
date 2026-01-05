"""Generate sample 'good' logs by simulating agent runs across test projects.

Usage:
    python scripts/generate_good_logs.py

This will run a deterministic sequence of tools for each test project and write
`simulated_session_<id>.jsonl` files into each project's `.coding_agent_logs`.
"""
import asyncio
import os
from pathlib import Path
import json

# Ensure OpenAI env vars are set for any LLM clients that initialize on import
os.environ.setdefault("OPENAI_API_KEY", os.getenv("CODING_AGENT_OPENAI_API_KEY", ""))
os.environ.setdefault("OPENAI_API_BASE", os.getenv("CODING_AGENT_OPENAI_API_BASE", ""))

from coding_agent.agent.session import InteractiveSession


async def run_for_project(project_path: Path):
    sess = InteractiveSession(str(project_path))
    entry = await sess.simulate_agent_run("Fix the failing tests")
    print(f"Wrote simulated log for {project_path}: {entry['timestamp']}")


def main():
    root = Path(__file__).resolve().parent.parent
    test_projects_dir = root / "test_projects"
    for proj in test_projects_dir.iterdir():
        if proj.is_dir():
            try:
                asyncio.run(run_for_project(proj))
            except Exception as e:
                print(f"Failed for {proj}: {e}")


if __name__ == '__main__':
    main()
