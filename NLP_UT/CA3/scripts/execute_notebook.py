#!/usr/bin/env python3
"""Execute a notebook cell-by-cell with visible progress logs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def _preview(source: str, max_len: int = 80) -> str:
    line = source.strip().splitlines()
    head = line[0] if line else ""
    return (head[: max_len - 3] + "...") if len(head) > max_len else head


def execute_notebook(path: Path, kernel_name: str) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name=kernel_name, allow_errors=False)

    code_cells = [i for i, c in enumerate(nb.cells) if c.get("cell_type") == "code"]
    total = len(code_cells)
    print(f"[notebook] Executing {path} with {total} code cells", flush=True)

    with client.setup_kernel():
        for order, idx in enumerate(code_cells, start=1):
            cell = nb.cells[idx]
            desc = _preview(cell.get("source", ""))
            print(f"[notebook] [{order}/{total}] cell {idx} start: {desc}", flush=True)
            t0 = time.time()
            client.execute_cell(cell, idx)
            dt = time.time() - t0
            print(f"[notebook] [{order}/{total}] cell {idx} done in {dt:.1f}s", flush=True)

    nbformat.write(nb, path)
    print(f"[notebook] Wrote executed notebook: {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()

    execute_notebook(args.notebook, args.kernel)


if __name__ == "__main__":
    main()
