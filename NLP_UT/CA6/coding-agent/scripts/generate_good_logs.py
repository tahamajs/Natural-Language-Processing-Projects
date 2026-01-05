"""Generate sample 'good' logs by simulating agent runs across test projects.

Usage:
    python scripts/generate_good_logs.py

This will run a deterministic sequence of tools for each test project and write
`simulated_session_<id>.jsonl` files into each project's `.coding_agent_logs`.
"""
import os
from pathlib import Path
import json

# Ensure OpenAI env vars are set for any LLM clients that initialize on import
os.environ.setdefault("OPENAI_API_KEY", os.getenv("CODING_AGENT_OPENAI_API_KEY", ""))
os.environ.setdefault("OPENAI_API_BASE", os.getenv("CODING_AGENT_OPENAI_API_BASE", ""))

from coding_agent.utils.log_generator import generate_all, aggregate_simulated_logs


def main(aggregate: bool = True, output: str | None = None):
    root = Path(__file__).resolve().parent.parent

    results = generate_all(root)

    # Print summary per project
    for r in results:
        if "error" in r:
            print(f"Failed for {r['project']}: {r['error']}")
        else:
            ts = r['entry'].get('timestamp') if isinstance(r.get('entry'), dict) else None
            print(f"Wrote simulated log for {r['project']}: {ts}")

    if aggregate:
        out = Path(output) if output else root / "simulated_logs_summary.jsonl"
        aggregate_simulated_logs(root, out)
        print(f"Aggregated simulated logs to: {out}")


if __name__ == '__main__':
    main()
