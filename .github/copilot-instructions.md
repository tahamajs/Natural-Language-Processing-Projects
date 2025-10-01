# Copilot Instructions for Natural-Language-Processing-Projects-SUT

## Repository shape
- Coursework archive anchored at `CAs/` (current Sharif assignments) and `nlp-projects/` (older ETH handins); each subfolder is self-contained with `code/`, `data/`, `model/`, and `report/` assets.
- Top-level notebooks live under `CAs/*/code/` and `CAs/*/report/`; the only standalone Python module is `nlp-projects/Assignment-1/backprop.py`.
- `tmp/` mirrors of `nlp-projects/` are read-only snapshots—do not modify or copy work there.

## Environment setup
- Use Python 3.9+ with `pip install -r requirements.txt`; this single file is a superset across assignments (PyTorch, gensim, hazm, scrapy).
- First-time setup needs NLTK corpora (`python - <<'PY'\nimport nltk; nltk.download('punkt')\nnltk.download('averaged_perceptron_tagger')\nPY`) and Hazm resources (`python -m hazm download resources`).
- Prefer Jupyter/VS Code notebooks; when scripting, keep per-project virtualenv kernels to avoid re-downloading large models.

## Per-project workflows
- **A1 Persian Poetry (CAs/A1-...)**: Scrapy spider defined inside `code/Report.ipynb`; web crawl results cached in `code/ganjoor.jsonlines`. Treat the crawl as read-only—network scraping is out of scope unless explicitly requested.
- **HeKasreh POS (CAs/HeKasreh-Analysis)**: Feature engineering and model training live in `code/report.ipynb`; pretrained `model/postagger.model` should be loaded, not regenerated, unless the user asks for retraining.
- **Medicine Name Extraction (CAs/Medicine-Name-Extraction)**: Alignment pipeline is notebook-driven; embeddings stored under `model/` are ~100MB. Stream from disk using gensim APIs, do not commit regenerated binaries.
- **Sentiment Analysis (CAs/Sentiment-Analysis)**: PyTorch LSTM workflow inside `code/report_rev1.ipynb`, with TensorBoard logs in `logs/`; reuse helper cells instead of creating new scripts.

## Coding conventions
- Stick to notebook cells for experiment code; if you must add Python modules, place them in `CAs/<project>/src/` and expose functions that notebooks can import.
- Data paths are relative to each assignment root (e.g., `data/movie_train.jsonl`, `model/postagger.model`); avoid hard-coding absolute paths.
- When manipulating Persian text, rely on Hazm’s `Normalizer`, `word_tokenize`, and stopword utilities as already demonstrated.

## Validation expectations
- No automated test suite; verification means rerunning the modified notebook sections and comparing metrics (accuracy/F1) logged in final cells.
- For PyTorch training runs, respect `logs/` layout so TensorBoard can read new experiment directories.

## Data and asset handling
- Large binaries (`*.model`, `*_embeddings.txt`, logs) are tracked—never rewrite in place without confirming with the user; stage new versions alongside originals (e.g., `postagger.model.v2`).
- CSV/JSONL datasets are canonical; any sampling should go to a new file under `data/derived/` to keep the raw splits intact.

## Common pitfalls
- Scrapy imports (`crochet`, `CrawlerRunner`) only exist inside notebooks; creating standalone scripts requires manual reactor management—prefer duplicating notebook patterns.
- TorchText versions in `requirements.txt` match PyTorch≥1.10; mixing newer TorchText APIs will break the notebooks—stay with `basic_english` tokenizer and manual vocab building already in place.
- Embedded English/Persian models assume UTF-8; when loading with gensim ensure `encoding='utf-8'` to avoid decoding errors on macOS.
