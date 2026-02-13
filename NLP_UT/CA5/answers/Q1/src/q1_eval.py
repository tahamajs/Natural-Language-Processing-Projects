from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from q1_pipeline import DEFAULT_INDEX_PATH, load_runtime_env, run_pipeline


DEFAULT_BENCHMARK_QUERIES = [
    {"question": "قانون کار درباره ساعات اضافه کاری چه می گوید؟", "ground_truth": ""},
    {"question": "شرایط صدور چک برگشتی چیست؟", "ground_truth": ""},
    {"question": "در قانون مدنی درباره اجاره چه آمده است؟", "ground_truth": ""},
    {"question": "جرایم مالیاتی شامل چه مواردی است؟", "ground_truth": ""},
    {"question": "سلام", "ground_truth": ""},
]


def _load_queries(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_BENCHMARK_QUERIES
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Benchmark input not found: {source}")
    if source.suffix.lower() in {".json"}:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Benchmark JSON must contain a list of records.")
        return [dict(item) for item in payload]
    if source.suffix.lower() in {".jsonl"}:
        rows: list[dict[str, str]] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(dict(json.loads(line)))
        return rows
    if source.suffix.lower() in {".csv"}:
        df = pd.read_csv(source)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported benchmark file extension: {source.suffix}")


def run_benchmark(
    input_records_path: str | Path | None,
    output_csv_path: str | Path,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    k: int = 10,
    top_n: int = 3,
) -> str:
    queries = _load_queries(input_records_path)
    rows: list[dict[str, Any]] = []
    for item in queries:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        result = run_pipeline(question, k=k, top_n=top_n, index_path=index_path)
        row: dict[str, Any] = {
            "question": question,
            "ground_truth": str(item.get("ground_truth", "")),
            "answer": result.get("answer", ""),
            "intent": result.get("intent", ""),
            "contexts": json.dumps(result.get("contexts", []), ensure_ascii=False),
            "context_count": len(result.get("contexts", [])),
            "hit_at_1": int(len(result.get("contexts", [])) >= 1),
            "hit_at_3": int(len(result.get("contexts", [])) >= 3),
        }
        for key, value in dict(result.get("timings", {})).items():
            row[key] = float(value)
        rows.append(row)
    if not rows:
        raise RuntimeError("No benchmark rows were generated.")
    df = pd.DataFrame(rows)
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)


def _require_env_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def run_ragas_evaluation(input_records_path: str, output_xlsx_path: str) -> str:
    load_runtime_env()
    _require_env_key("OPENAI_API_KEY")
    openai_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model_name = os.getenv("EVAL_OPENAI_MODEL", "gpt-4o-mini")
    embedding_model_name = os.getenv("EVAL_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    benchmark_df = pd.read_csv(input_records_path)
    if benchmark_df.empty:
        raise RuntimeError("Benchmark CSV is empty.")

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    contexts: list[list[str]] = []
    for payload in benchmark_df["contexts"].fillna("[]").tolist():
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = []
        if not isinstance(parsed, list):
            parsed = []
        normalized_contexts = []
        for item in parsed:
            if isinstance(item, dict):
                normalized_contexts.append(str(item.get("text", "")))
            else:
                normalized_contexts.append(str(item))
        contexts.append(normalized_contexts)

    ragas_input = Dataset.from_dict(
        {
            "question": benchmark_df["question"].fillna("").astype(str).tolist(),
            "answer": benchmark_df["answer"].fillna("").astype(str).tolist(),
            "contexts": contexts,
            "ground_truth": benchmark_df["ground_truth"].fillna("").astype(str).tolist(),
        }
    )

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=openai_base,
    )
    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=openai_base,
    )

    wrapped_llm = LangchainLLMWrapper(llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    evaluation = evaluate(
        dataset=ragas_input,
        metrics=[faithfulness, answer_relevancy],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings,
    )

    evaluation_df = evaluation.to_pandas()
    metric_columns = [column for column in ["faithfulness", "answer_relevancy"] if column in evaluation_df.columns]
    if metric_columns and bool(evaluation_df[metric_columns].isna().all().all()):
        raise RuntimeError(
            "RAGAS evaluation returned no valid metric values. Check OPENAI_API_KEY and OPENAI_API_BASE."
        )
    merged = pd.concat([benchmark_df.reset_index(drop=True), evaluation_df.reset_index(drop=True)], axis=1)

    output = Path(output_xlsx_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(output, index=False)
    return str(output)
