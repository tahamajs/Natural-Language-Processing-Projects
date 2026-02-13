from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)


def generate_all(input_csv_path: str, output_dir: str) -> list[str]:
    source = Path(input_csv_path)
    if not source.exists():
        raise FileNotFoundError(f"Scenario CSV not found: {source}")

    df = pd.read_csv(source)
    if df.empty:
        raise RuntimeError("Scenario CSV is empty.")

    sns.set_theme(style="whitegrid")
    out_dir = Path(output_dir)
    generated: list[str] = []

    plt.figure(figsize=(10, 4))
    sns.barplot(x="scenario_id", y="latency_seconds", data=df, color="#2563eb")
    plt.title("Scenario Latency")
    plt.ylabel("Seconds")
    plt.xlabel("Scenario")
    generated.append(_save(out_dir / "q2_scenario_latency.png"))

    plt.figure(figsize=(8, 4))
    sns.histplot(df["latency_seconds"].dropna(), bins=15, kde=True, color="#0f766e")
    plt.title("Latency Distribution")
    plt.xlabel("Seconds")
    generated.append(_save(out_dir / "q2_latency_distribution.png"))

    exploded = []
    for names in df["tool_names"].fillna("").astype(str).tolist():
        for name in [item.strip() for item in names.split(",") if item.strip()]:
            exploded.append(name)
    if exploded:
        usage = pd.Series(exploded).value_counts().reset_index()
        usage.columns = ["tool", "count"]
        plt.figure(figsize=(10, 4))
        sns.barplot(x="tool", y="count", data=usage, palette="crest")
        plt.xticks(rotation=30, ha="right")
        plt.title("Tool Usage Frequency")
        generated.append(_save(out_dir / "q2_tool_usage_frequency.png"))

    if "query" in df.columns:
        temp = df.copy()
        temp["query_length"] = temp["query"].fillna("").astype(str).str.len()
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=temp, x="query_length", y="latency_seconds", alpha=0.8)
        plt.title("Query Length vs Latency")
        generated.append(_save(out_dir / "q2_query_length_vs_latency.png"))

    return generated
