from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REQUIRED_ENV_KEYS = [
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "AMADEUS_CLIENT_ID",
    "AMADEUS_CLIENT_SECRET",
    "TAVILY_API_KEY",
    "LLAMA_CLOUD_API_KEY",
]

REQUIRED_COMMANDS = ["jupyter", "xelatex", "tesseract"]

REQUIRED_MODULES = [
    "sklearn",
    "openpyxl",
    "ragas",
    "datasets",
    "langchain_openai",
    "chainlit",
    "wordcloud",
    "pytesseract",
    "fitz",
    "lancedb",
    "sentence_transformers",
    "amadeus",
    "tavily",
    "kagglehub",
    "llama_parse",
    "langgraph",
    "gdown",
]

PLACEHOLDER_TOKENS = {
    "your_api_key_here",
    "your_openai_api_key_here",
    "your_openai_base_here",
    "your_amadeus_client_id_here",
    "your_amadeus_client_secret_here",
    "your_tavily_api_key_here",
    "your_llama_cloud_api_key_here",
    "replace_me",
    "changeme",
    "dummy",
}


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


def _load_runtime_env() -> None:
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


def _check_commands(errors: list[str]) -> None:
    for command in REQUIRED_COMMANDS:
        if shutil.which(command) is None:
            errors.append(f"Missing required command: {command}")


def _check_nbconvert(errors: list[str]) -> None:
    completed = subprocess.run(
        ["jupyter", "nbconvert", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        errors.append("jupyter nbconvert is not available")


def _check_tesseract_fas(errors: list[str]) -> None:
    completed = subprocess.run(
        ["tesseract", "--list-langs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        errors.append("tesseract --list-langs failed")
        return
    langs = {line.strip() for line in completed.stdout.splitlines()}
    if "fas" not in langs:
        errors.append("Persian OCR language 'fas' is not installed in tesseract")


def _check_modules(errors: list[str]) -> None:
    for module in REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None:
            errors.append(f"Missing required Python module: {module}")


def _check_env(strict_live: bool, errors: list[str]) -> None:
    if not strict_live:
        return
    for key in REQUIRED_ENV_KEYS:
        value = os.getenv(key, "").strip()
        if not value:
            errors.append(f"Missing required environment variable: {key}")
            continue
        normalized = value.lower()
        if normalized in PLACEHOLDER_TOKENS or (
            normalized.startswith("your_") and normalized.endswith("_here")
        ):
            errors.append(f"Environment variable appears to be a placeholder: {key}")


def _check_openai_credentials(strict_live: bool, errors: list[str]) -> None:
    if not strict_live:
        return
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip().rstrip("/")
    if not api_key or not api_base:
        return
    request = Request(
        f"{api_base}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=15) as response:
            code = getattr(response, "status", 200)
            if code >= 400:
                errors.append(f"OpenAI API credential check failed with status {code}")
    except HTTPError as exc:
        errors.append(f"OpenAI API credential check failed with status {exc.code}")
    except URLError as exc:
        errors.append(f"OpenAI API credential check failed: {exc.reason}")
    except Exception as exc:
        errors.append(f"OpenAI API credential check failed: {type(exc).__name__}")


def check_all(strict_live: bool = True) -> int:
    _load_runtime_env()
    errors: list[str] = []
    _check_commands(errors)
    if not errors:
        _check_nbconvert(errors)
    _check_tesseract_fas(errors)
    _check_modules(errors)
    _check_env(strict_live, errors)
    _check_openai_credentials(strict_live, errors)

    if errors:
        print("Preflight failed")
        for issue in errors:
            print(f"- {issue}")
        return 1

    print("Preflight passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-live", action="store_true", default=False)
    args = parser.parse_args()
    return check_all(strict_live=args.strict_live)


if __name__ == "__main__":
    raise SystemExit(main())
