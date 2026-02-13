from __future__ import annotations

import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_INDEX_PATH = Path("answers/Q1/data/indexed_dataset.pkl")
GREETING_KEYWORDS = ["سلام", "درود", "صبح بخیر", "وقت بخیر", "hello", "hi"]
ABUSE_KEYWORDS = ["مسخره", "بی‌ادب", "احمق", "نادان", "stupid", "idiot"]


def _parse_env_line(line: str) -> tuple[str, str] | None:
    clean = line.strip()
    if not clean or clean.startswith("#") or "=" not in clean:
        return None
    key, value = clean.split("=", 1)
    key = key.strip()
    value = value.strip().strip('"').strip("'")
    if not key:
        return None
    return key, value


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if os.environ.get(key):
            continue
        os.environ[key] = value


def _load_json_keys(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if os.environ.get(key):
            continue
        os.environ[key] = str(value)


def load_runtime_env() -> None:
    candidates = [
        Path("answers/Q1/.env"),
        Path("answers/Q1/.keys.json"),
        Path("answers/Q2/.env"),
        Path("answers/Q2/.keys.json"),
        Path(".env"),
        Path(".keys.json"),
    ]
    for candidate in candidates:
        if candidate.suffix == ".json":
            _load_json_keys(candidate)
        else:
            _load_env_file(candidate)


def _normalize_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("ي", "ی").replace("ك", "ک")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _split_text(text: str, max_chars: int = 1200, overlap: int = 120) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _extract_text_native(pdf_path: Path) -> str:
    import fitz

    doc = fitz.open(pdf_path)
    pages: list[str] = []
    for idx in range(len(doc)):
        pages.append(doc.load_page(idx).get_text("text"))
    doc.close()
    return "\n".join(pages)


def _extract_text_ocr(pdf_path: Path, lang: str = "fas") -> str:
    import fitz
    import pytesseract
    from PIL import Image

    doc = fitz.open(pdf_path)
    pages: list[str] = []
    for idx in range(len(doc)):
        page = doc.load_page(idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(pytesseract.image_to_string(image, lang=lang))
    doc.close()
    return "\n".join(pages)


def extract_pdf_text(pdf_path: Path) -> str:
    native_text = _extract_text_native(pdf_path)
    if len(_normalize_text(native_text)) >= 500:
        return native_text
    try:
        ocr_text = _extract_text_ocr(pdf_path, lang="fas")
    except Exception:
        ocr_text = ""
    return ocr_text or native_text


def _title_from_filename(filename: str) -> str:
    title = Path(filename).stem
    title = title.replace("نسخه چاپی", "").replace("_page-0001", "")
    title = title.replace("_", " ").strip()
    return title or Path(filename).stem


def _guess_year(text: str) -> int | None:
    match = re.search(r"(?:13|14)\d{2}", text)
    if not match:
        return None
    return int(match.group(0))


def _collect_documents(input_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    pdf_files = sorted(input_dir.glob("*.pdf"))
    docs: list[str] = []
    metadata: list[dict[str, Any]] = []
    doc_id = 0
    for pdf_file in pdf_files:
        raw_text = extract_pdf_text(pdf_file)
        chunks = _split_text(raw_text)
        base_title = _title_from_filename(pdf_file.name)
        for chunk_index, chunk in enumerate(chunks):
            year = _guess_year(chunk)
            docs.append(chunk)
            metadata.append(
                {
                    "doc_id": doc_id,
                    "title": f"{base_title} - Part {chunk_index + 1}",
                    "filename": pdf_file.name,
                    "chunk_index": chunk_index,
                    "year": year,
                    "length": len(chunk),
                }
            )
            doc_id += 1
    return docs, metadata


def build_index(input_dir: str, output_path: str) -> str:
    input_path = Path(input_dir)
    output = Path(output_path)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    docs, metadata = _collect_documents(input_path)
    if not docs:
        raise RuntimeError(f"No documents found under: {input_path}")

    vectorizer = TfidfVectorizer(
        preprocessor=_normalize_text,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        lowercase=False,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(docs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(
            {
                "docs": docs,
                "metadata": metadata,
                "vectorizer": vectorizer,
                "X": matrix,
            },
            handle,
        )
    return str(output)


def load_index(index_path: str | Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    with path.open("rb") as handle:
        store = pickle.load(handle)
    return store


def rewrite_query(query: str) -> str:
    return _normalize_text(query)


def classify_intent(query: str) -> dict[str, str]:
    lowered = query.lower()
    for keyword in GREETING_KEYWORDS:
        if keyword in lowered:
            return {"intent": "greeting", "response": "Hello. Ask any legal question."}
    for keyword in ABUSE_KEYWORDS:
        if keyword in lowered:
            return {
                "intent": "abuse",
                "response": "Please ask politely so I can help accurately.",
            }
    return {"intent": "law_question", "response": ""}


def extract_metadata(rewritten_query: str) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    year_match = re.search(r"(?:13|14)\d{2}|\b\d{4}\b", rewritten_query)
    if year_match:
        extracted["year"] = int(year_match.group(0))
    keyword_candidates = ["کار", "مدنی", "جزا", "مالیات", "تامین", "املاک", "چک"]
    keywords = [word for word in keyword_candidates if word in rewritten_query]
    if keywords:
        extracted["keywords"] = keywords
    return extracted


def _match_metadata(meta: dict[str, Any], text: str, metadata_filter: dict[str, Any]) -> bool:
    if "year" in metadata_filter:
        if meta.get("year") != metadata_filter["year"]:
            return False
    if "keywords" in metadata_filter:
        title = str(meta.get("title", ""))
        lowered_text = str(text)
        if not any(keyword in title or keyword in lowered_text for keyword in metadata_filter["keywords"]):
            return False
    return True


def context_retrieve(
    rewritten_query: str,
    metadata_filter: dict[str, Any] | None = None,
    k: int = 10,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> list[dict[str, Any]]:
    store = load_index(index_path)
    docs = store["docs"]
    metadata = store["metadata"]
    vectorizer = store["vectorizer"]
    matrix = store["X"]

    query_vector = vectorizer.transform([rewritten_query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked = np.argsort(-scores)

    contexts: list[dict[str, Any]] = []
    for index in ranked:
        doc_text = docs[int(index)]
        doc_meta = metadata[int(index)]
        if metadata_filter and not _match_metadata(doc_meta, doc_text, metadata_filter):
            continue
        contexts.append(
            {
                "doc_id": int(index),
                "text": doc_text,
                "metadata": doc_meta,
                "score": float(scores[int(index)]),
            }
        )
        if len(contexts) >= k:
            break
    return contexts


def rerank(contexts: list[dict[str, Any]], top_n: int = 3, relevance_threshold: float = 0.01) -> list[dict[str, Any]]:
    if not contexts:
        return []
    ordered = sorted(contexts, key=lambda item: item["score"], reverse=True)[:top_n]
    if all(item["score"] < relevance_threshold for item in ordered):
        return []
    return ordered


def generate_answer(rewritten_query: str, contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "No relevant evidence was retrieved. Please ask with more specific legal terms."
    blocks: list[str] = []
    for rank, context in enumerate(contexts, start=1):
        title = str(context["metadata"].get("title", "Unknown source"))
        snippet = str(context["text"])[:420].strip()
        blocks.append(
            f"Source {rank}: {title}\nScore: {context['score']:.4f}\nEvidence: {snippet}"
        )
    return "\n\n".join(blocks)


def run_pipeline(query: str, k: int = 10, top_n: int = 3, index_path: str | Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    load_runtime_env()
    started = time.time()
    timings: dict[str, float] = {}

    step_start = time.time()
    rewritten = rewrite_query(query)
    timings["rewrite"] = time.time() - step_start

    step_start = time.time()
    intent = classify_intent(rewritten)
    timings["classify_intent"] = time.time() - step_start

    if intent["intent"] != "law_question":
        timings["total"] = time.time() - started
        return {
            "answer": intent["response"],
            "timings": timings,
            "contexts": [],
            "intent": intent["intent"],
            "rewritten": rewritten,
            "meta_filter": {},
        }

    step_start = time.time()
    metadata_filter = extract_metadata(rewritten)
    timings["extract_metadata"] = time.time() - step_start

    step_start = time.time()
    contexts = context_retrieve(rewritten, metadata_filter, k=k, index_path=index_path)
    timings["context_retrieve"] = time.time() - step_start

    step_start = time.time()
    top_contexts = rerank(contexts, top_n=top_n)
    if not top_contexts:
        contexts = context_retrieve(rewritten, None, k=max(top_n * 3, k), index_path=index_path)
        top_contexts = rerank(contexts, top_n=top_n)
    timings["rerank"] = time.time() - step_start

    step_start = time.time()
    answer = generate_answer(rewritten, top_contexts)
    timings["generate_answer"] = time.time() - step_start

    timings["total"] = time.time() - started
    return {
        "answer": answer,
        "timings": timings,
        "contexts": top_contexts,
        "intent": intent["intent"],
        "rewritten": rewritten,
        "meta_filter": metadata_filter,
    }
