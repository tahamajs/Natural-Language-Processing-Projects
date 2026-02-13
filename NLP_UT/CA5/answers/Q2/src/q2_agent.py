from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import kagglehub
import lancedb
import pandas as pd
from amadeus import Client, ResponseError
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from llama_parse import LlamaParse
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient


REQUIRED_KEYS = [
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "AMADEUS_CLIENT_ID",
    "AMADEUS_CLIENT_SECRET",
    "TAVILY_API_KEY",
    "LLAMA_CLOUD_API_KEY",
]


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
        Path("answers/Q2/.env"),
        Path("answers/Q2/.keys.json"),
        Path("answers/Q1/.env"),
        Path("answers/Q1/.keys.json"),
        Path(".env"),
        Path(".keys.json"),
    ]
    for candidate in candidates:
        if candidate.suffix == ".json":
            _load_json_keys(candidate)
        else:
            _load_env_file(candidate)


def _require_keys() -> None:
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key, "").strip()]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {
            "faq_js_path": "answers/Q2/FAQ.js",
            "tourism_pdf_path": "answers/Q2/World Travel Book.pdf",
            "lancedb_dir": "answers/Q2/lancedb_data",
            "openai_model": "gpt-4o-mini",
            "openai_temperature": 0,
            "max_tool_results": 5,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Q2 config must be a JSON object.")
    defaults = {
        "faq_js_path": "answers/Q2/FAQ.js",
        "tourism_pdf_path": "answers/Q2/World Travel Book.pdf",
        "lancedb_dir": "answers/Q2/lancedb_data",
        "openai_model": "gpt-4o-mini",
        "openai_temperature": 0,
        "max_tool_results": 5,
    }
    defaults.update(payload)
    return defaults


def _read_first_csv(path: str | Path) -> pd.DataFrame:
    root = Path(path)
    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in dataset path: {root}")
    return pd.read_csv(csv_files[0])


def _build_iata_map() -> dict[str, str]:
    path = kagglehub.dataset_download("zinovadr/iata-airport-code")
    df = _read_first_csv(path)
    normalized_columns = {col.lower(): col for col in df.columns}
    city_col = None
    code_col = None
    for candidate in ["city", "municipality", "name"]:
        if candidate in normalized_columns:
            city_col = normalized_columns[candidate]
            break
    for candidate in ["iata_code", "iata", "iata code"]:
        if candidate in normalized_columns:
            code_col = normalized_columns[candidate]
            break
    if city_col is None or code_col is None:
        raise RuntimeError("IATA dataset does not include required city/code columns.")

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        city = str(row.get(city_col, "")).strip().lower()
        code = str(row.get(code_col, "")).strip().upper()
        if not city or not code or code == "NAN":
            continue
        mapping[city] = code
    return mapping


def _build_currency_map() -> dict[str, str]:
    path = kagglehub.dataset_download("phanee16/currency-and-country-code-mapping")
    df = _read_first_csv(path)
    normalized = {col.lower(): col for col in df.columns}
    country_col = None
    code_col = None
    for candidate in ["country", "country_name", "name"]:
        if candidate in normalized:
            country_col = normalized[candidate]
            break
    for candidate in ["code", "currency_code", "currency"]:
        if candidate in normalized:
            code_col = normalized[candidate]
            break
    if country_col is None or code_col is None:
        raise RuntimeError("Currency dataset does not include required country/code columns.")

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        country = str(row.get(country_col, "")).strip().lower()
        code = str(row.get(code_col, "")).strip().upper()
        if not country or not code or code == "NAN":
            continue
        mapping[country] = code
    return mapping


def _load_faq_data(js_path: Path) -> list[dict[str, str]]:
    if not js_path.exists():
        raise FileNotFoundError(f"FAQ file not found: {js_path}")
    content = js_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
    if not match:
        raise RuntimeError("FAQ.js does not contain a valid JSON array payload.")
    payload = json.loads(match.group(0))
    if not isinstance(payload, list):
        raise RuntimeError("Parsed FAQ payload is not a list.")
    normalized: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            normalized.append({"question": question, "answer": answer})
    if not normalized:
        raise RuntimeError("FAQ dataset is empty after normalization.")
    return normalized


def _parse_tourism_book(pdf_path: Path) -> list[str]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Tourism PDF not found: {pdf_path}")
    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], result_type="markdown")
    docs = parser.load_data(str(pdf_path))
    texts = [str(getattr(doc, "text", "")).strip() for doc in docs]
    texts = [text for text in texts if text]
    if not texts:
        raise RuntimeError("No text extracted from tourism PDF.")
    return texts


def _embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def _init_vector_db(lancedb_dir: Path, faq_rows: list[dict[str, str]], tourism_sections: list[str]) -> dict[str, Any]:
    lancedb_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    dimension = int(model.get_sentence_embedding_dimension())

    faq_questions = [row["question"] for row in faq_rows]
    faq_vectors = _embed_texts(model, faq_questions)
    faq_table_rows = [
        {
            "id": idx,
            "question": faq_rows[idx]["question"],
            "answer": faq_rows[idx]["answer"],
            "vector": faq_vectors[idx],
        }
        for idx in range(len(faq_rows))
    ]

    tourism_vectors = _embed_texts(model, tourism_sections)
    tourism_rows = [
        {
            "id": idx,
            "text": tourism_sections[idx],
            "vector": tourism_vectors[idx],
        }
        for idx in range(len(tourism_sections))
    ]

    db = lancedb.connect(str(lancedb_dir))
    faq_table = db.create_table("faq_table", data=faq_table_rows, mode="overwrite")
    tourism_table = db.create_table("tourism_table", data=tourism_rows, mode="overwrite")
    return {
        "db": db,
        "faq_table": faq_table,
        "tourism_table": tourism_table,
        "embedding_model": model,
        "embedding_dim": dimension,
    }


def _city_to_iata(city: str, iata_map: dict[str, str]) -> str:
    key = str(city).strip().lower()
    if key in iata_map:
        return iata_map[key]
    for known_city, code in iata_map.items():
        if key in known_city or known_city in key:
            return code
    return ""


def _country_to_currency(country: str, currency_map: dict[str, str]) -> str:
    key = str(country).strip().lower()
    if key in currency_map:
        return currency_map[key]
    for known_country, code in currency_map.items():
        if key in known_country or known_country in key:
            return code
    return ""


def build_agent(config_path: str):
    load_runtime_env()
    _require_keys()

    config = _load_config(config_path)
    faq_path = Path(str(config["faq_js_path"]))
    tourism_pdf = Path(str(config["tourism_pdf_path"]))
    lancedb_dir = Path(str(config["lancedb_dir"]))
    max_results = int(config.get("max_tool_results", 5))

    faq_rows = _load_faq_data(faq_path)
    tourism_sections = _parse_tourism_book(tourism_pdf)
    vector_state = _init_vector_db(lancedb_dir, faq_rows, tourism_sections)
    iata_map = _build_iata_map()
    currency_map = _build_currency_map()

    amadeus = Client(
        client_id=os.environ["AMADEUS_CLIENT_ID"],
        client_secret=os.environ["AMADEUS_CLIENT_SECRET"],
    )
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    faq_table = vector_state["faq_table"]
    tourism_table = vector_state["tourism_table"]
    embedding_model: SentenceTransformer = vector_state["embedding_model"]

    @tool
    def flight_search_tool(origin_city: str, destination_city: str, departure_date: str) -> str:
        origin_code = _city_to_iata(origin_city, iata_map)
        destination_code = _city_to_iata(destination_city, iata_map)
        if not origin_code or not destination_code:
            return "IATA code mapping failed for one or both cities."
        try:
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin_code,
                destinationLocationCode=destination_code,
                departureDate=departure_date,
                adults=1,
                max=max_results,
            )
        except ResponseError as exc:
            return f"Amadeus flight API failed: {exc}"

        offers = response.data or []
        if not offers:
            return "No flight offers returned by Amadeus."

        lines: list[str] = []
        for idx, offer in enumerate(offers, start=1):
            price = str(offer.get("price", {}).get("total", "N/A"))
            currency = str(offer.get("price", {}).get("currency", ""))
            itinerary = offer.get("itineraries", [{}])[0]
            segments = itinerary.get("segments", [])
            if segments:
                carrier = segments[0].get("carrierCode", "")
                dep = segments[0].get("departure", {}).get("at", "")
                arr = segments[-1].get("arrival", {}).get("at", "")
                duration = itinerary.get("duration", "")
                lines.append(
                    f"Option {idx}: carrier={carrier}, price={price} {currency}, departure={dep}, arrival={arr}, duration={duration}"
                )
        return "\n".join(lines) if lines else "No parsable flight offers were returned."

    @tool
    def hotel_search_tool(city: str, check_in_date: str, check_out_date: str) -> str:
        city_code = _city_to_iata(city, iata_map)
        if not city_code:
            return "City IATA code not found for hotel search."
        try:
            response = amadeus.shopping.hotel_offers_search.get(
                cityCode=city_code,
                checkInDate=check_in_date,
                checkOutDate=check_out_date,
                adults=1,
                max=max_results,
            )
            offers = response.data or []
            if not offers:
                return "No hotel offers returned by Amadeus."
            lines: list[str] = []
            for idx, offer in enumerate(offers, start=1):
                hotel_name = str(offer.get("hotel", {}).get("name", "Unknown"))
                first_offer = offer.get("offers", [{}])[0]
                total = str(first_offer.get("price", {}).get("total", "N/A"))
                currency = str(first_offer.get("price", {}).get("currency", ""))
                lines.append(f"Option {idx}: hotel={hotel_name}, total={total} {currency}")
            return "\n".join(lines)
        except ResponseError as exc:
            return f"Amadeus hotel API failed: {exc}"

    @tool
    def restaurant_search_tool(destination: str) -> str:
        query = f"best top rated restaurants in {destination} with cuisine and price"
        response = tavily_client.search(query=query, max_results=max_results)
        rows = response.get("results", [])
        if not rows:
            return "No restaurant results returned by Tavily."
        lines: list[str] = []
        for idx, row in enumerate(rows, start=1):
            title = str(row.get("title", "Untitled"))
            content = str(row.get("content", "")).strip().replace("\n", " ")
            url = str(row.get("url", ""))
            lines.append(f"Result {idx}: {title} | {content[:220]} | {url}")
        return "\n".join(lines)

    @tool
    def weather_info_tool(destination: str, date: str) -> str:
        query = f"weather forecast in {destination} on {date} temperature precipitation"
        response = tavily_client.search(query=query, max_results=max_results)
        rows = response.get("results", [])
        if not rows:
            return "No weather results returned by Tavily."
        lines: list[str] = []
        for idx, row in enumerate(rows, start=1):
            title = str(row.get("title", "Untitled"))
            content = str(row.get("content", "")).strip().replace("\n", " ")
            lines.append(f"Result {idx}: {title} | {content[:220]}")
        return "\n".join(lines)

    @tool
    def currency_info_tool(source_country: str, target_country: str) -> str:
        source_code = _country_to_currency(source_country, currency_map)
        target_code = _country_to_currency(target_country, currency_map)
        if not source_code or not target_code:
            return "Currency code mapping failed for one or both countries."
        query = f"{source_code} to {target_code} exchange rate today"
        response = tavily_client.search(query=query, max_results=max_results)
        rows = response.get("results", [])
        if not rows:
            return "No exchange-rate results returned by Tavily."
        best = rows[0]
        return f"{source_code} to {target_code}: {best.get('content', '')}"

    @tool
    def faq_search_tool(question: str) -> str:
        vector = embedding_model.encode([question], normalize_embeddings=True)[0].tolist()
        rows = faq_table.search(vector).limit(max_results).to_list()
        if not rows:
            return "No FAQ matches found."
        lines: list[str] = []
        for idx, row in enumerate(rows, start=1):
            lines.append(
                f"Match {idx}: question={row.get('question', '')} | answer={row.get('answer', '')}"
            )
        return "\n".join(lines)

    @tool
    def trip_planning_tool(destination: str, days: int) -> str:
        vector = embedding_model.encode([destination], normalize_embeddings=True)[0].tolist()
        sections = tourism_table.search(vector).limit(max_results).to_list()
        content = "\n".join(str(item.get("text", ""))[:350] for item in sections)
        weather = weather_info_tool.invoke({"destination": destination, "date": "today"})
        restaurants = restaurant_search_tool.invoke({"destination": destination})
        itinerary_prompt = (
            f"Create a {days}-day itinerary for {destination}. "
            f"Use this tourism context: {content}. "
            f"Use this weather context: {weather}. "
            f"Use these restaurant options: {restaurants}."
        )
        llm = ChatOpenAI(
            model=str(config["openai_model"]),
            temperature=float(config.get("openai_temperature", 0)),
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base=os.environ["OPENAI_API_BASE"],
        )
        response = llm.invoke([HumanMessage(content=itinerary_prompt)])
        return str(response.content)

    tools = [
        flight_search_tool,
        hotel_search_tool,
        restaurant_search_tool,
        weather_info_tool,
        currency_info_tool,
        faq_search_tool,
        trip_planning_tool,
    ]

    llm = ChatOpenAI(
        model=str(config["openai_model"]),
        temperature=float(config.get("openai_temperature", 0)),
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"],
    )

    prompt = (
        "You are TravBot, a travel operations assistant. "
        "Always select tools when user intent requires external travel data. "
        "Return concise factual answers and include key assumptions."
    )

    agent = create_react_agent(llm, tools, prompt=prompt)
    return agent
