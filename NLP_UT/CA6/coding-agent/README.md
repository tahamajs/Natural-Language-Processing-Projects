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

Generated logs are written to `.coding_agent_logs/simulated_session_*.jsonl` in each project folder, and an aggregated file `simulated_logs_summary.jsonl` will be created at the repository root by default.
