# Coding Agent

This repository contains a CLI-based coding agent for the assignment. New features added:

- `generate_logs` command: Generate simulated 'good' logs for included test projects and optionally aggregate them to a single summary file.

Usage:

- Generate logs and aggregate to root summary (JSONL):

  coding-agent generate-logs

- Generate logs without aggregation:

  coding-agent generate-logs --no-aggregate

- Generate logs and specify aggregate output path:

  coding-agent generate-logs --output /path/to/my_summary.jsonl

- Export aggregated summary directly as CSV or Markdown:

  coding-agent generate-logs --format csv
  coding-agent generate-logs --format md

- Control whether to include full file contents in logs (default: include full contents):

  coding-agent generate-logs --no-full  # will use truncation instead of full content

Generated logs are written to `.coding_agent_logs/simulated_session_*.jsonl` in each project folder, and an aggregated file `simulated_logs_summary.jsonl` will be created at the repository root by default. Use `--format` to export `simulated_logs_summary.csv` or `simulated_logs_summary.md` instead of JSONL.
