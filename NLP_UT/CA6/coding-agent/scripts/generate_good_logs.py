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


def main(aggregate: bool = True, output: str | None = None, output_format: str = "jsonl", full: bool = True):
    root = Path(__file__).resolve().parent.parent

    results = generate_all(root, full=full)

    # Print summary per project
    for r in results:
        if "error" in r:
            print(f"Failed for {r['project']}: {r['error']}")
        else:
            ts = r['entry'].get('timestamp') if isinstance(r.get('entry'), dict) else None
            print(f"Wrote simulated log for {r['project']}: {ts}")

    out = Path(output) if output else root / "simulated_logs_summary.jsonl"

    if aggregate:
        aggregate_simulated_logs(root, out)
        print(f"Aggregated simulated logs to: {out}")

    # If requested, convert aggregated JSONL to csv or markdown
    fmt = output_format.lower() if output_format else "jsonl"
    if fmt == "csv":
        from coding_agent.utils.log_generator import jsonl_to_csv

        csv_out = out.with_suffix(".csv")
        jsonl_to_csv(out, csv_out)
        print(f"Exported summary CSV to: {csv_out}")
    elif fmt in {"md", "markdown"}:
        from coding_agent.utils.log_generator import jsonl_to_markdown

        md_out = out.with_suffix(".md")
        jsonl_to_markdown(out, md_out)
        print(f"Exported summary Markdown to: {md_out}")
    else:
        print("No extra export requested or unrecognized format; leaving JSONL as-is.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate simulated agent logs for test projects')
    parser.add_argument('--no-aggregate', action='store_true', help='Do not aggregate generated logs into a single summary')
    parser.add_argument('--output', type=str, default=None, help='Output path for aggregated summary (JSONL)')
    parser.add_argument('--format', type=str, choices=['jsonl', 'csv', 'md'], default='jsonl', help='Export format for aggregated summary')
    parser.add_argument('--no-full', action='store_true', help='Do not include full file contents in generated logs (use truncation)')

    args = parser.parse_args()
    main(aggregate=not args.no_aggregate, output=args.output, output_format=args.format, full=not args.no_full)
