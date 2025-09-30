# Natural Language Processing Projects - Sharif University of Technology

This repository contains a collection of Natural Language Processing (NLP) projects completed as assignments for the NLP course at Sharif University of Technology. Each project demonstrates different aspects of NLP techniques, from text preprocessing and analysis to machine learning and deep learning applications.

## Course Information

- **Institution**: Sharif University of Technology
- **Course**: Natural Language Processing
- **Term**: Spring 1402 (2023-2024)
- **Instructor**: [Instructor Name]

## Projects Overview

### A1 - Persian Poetry Comments on Ganjoor

**Topic**: Analysis of User Comments on Persian Poetry

This project focuses on extracting and analyzing user comments from the Ganjoor website, a comprehensive online repository of Persian poetry. The system employs web scraping techniques to collect comments on works by renowned Persian poets including Omar Khayyam, Hafez, Saadi, Ferdousi, Nezami, and Moulavi.

**Key Features**:

- Web scraping using Scrapy framework
- Persian text preprocessing with Hazm library
- Text normalization, tokenization, and stopword removal
- Frequency analysis and poet popularity assessment
- Cross-reference analysis between poets

**Technologies**: Python, Scrapy, Hazm, NLTK, Pandas, JSONLines

[View detailed README](./A1-Persian-Poetry-Comments-on-Ganjoor/README.md)

---

### A2 - HeKasreh (Persian POS Tagging)

**Topic**: Part-of-Speech Tagging for Persian Language

This project implements a comprehensive Part-of-Speech (POS) tagging system specifically designed for the Persian language, with special attention to "He Kasreh" (ه کسره) morphological corrections. The system uses machine learning approaches to automatically assign grammatical tags to Persian words.

**Key Features**:

- Persian text preprocessing and normalization
- Feature engineering for POS tagging
- Multiple machine learning algorithms (Random Forest, etc.)
- Morphological analysis and "He Kasreh" corrections
- Model evaluation and performance metrics

**Technologies**: Python, NLTK, scikit-learn, Pandas, NumPy

[View detailed README](./A2-HeKasreh/README.md)

---

### A3 - Medicine Name Extraction

**Topic**: Cross-Lingual Medicine Name Extraction from Text

This project develops a system for extracting medicine names from multilingual textual data using advanced NLP techniques. It leverages word embeddings and cross-lingual alignment to identify medication names in both English and Persian texts, integrating with official drug databases.

**Key Features**:

- Word embedding training (Skip-gram models)
- Cross-lingual embedding alignment
- Similarity-based entity extraction
- Integration with WHO ATC and Iran FDA databases
- Support for English and Persian text processing

**Technologies**: Python, Gensim, scikit-learn, Pandas, NumPy

[View detailed README](./A3-MedicineNameExtraction/README.md)

---

### A4 - Sentiment Analysis

**Topic**: Deep Learning-based Sentiment Classification on Movie Reviews

This project implements a comprehensive sentiment analysis system using deep learning techniques to classify movie reviews as positive or negative. The system employs Recurrent Neural Networks with LSTM units for effective sequential text processing.

**Key Features**:

- LSTM-based sentiment classification model
- Comprehensive text preprocessing pipeline
- PyTorch implementation with proper data handling
- TensorBoard integration for training monitoring
- Extensive evaluation metrics and analysis

**Technologies**: Python, PyTorch, TorchText, Pandas, NumPy, scikit-learn, TensorBoard

[View detailed README](./A4-SentimentAnalysis/README.md)

## Common Technologies Used

- **Programming Language**: Python 3.x
- **NLP Libraries**: NLTK, Hazm, DadmaTools
- **Machine Learning**: scikit-learn, PyTorch
- **Data Processing**: Pandas, NumPy
- **Web Scraping**: Scrapy
- **Word Embeddings**: Gensim
- **Visualization**: Matplotlib, Seaborn

## Project Structure

```
nlp-assignments-spring-2023/
├── README.md                           # Main repository README
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore rules
├── A1-Persian-Poetry-Comments-on-Ganjoor/
│   ├── README.md
│   ├── code/
│   │   ├── ganjoor.jsonlines
│   │   └── Report.ipynb
│   ├── report/
│   └── assets/
├── A2-HeKasreh/
│   ├── README.md
│   ├── code/
│   ├── data/
│   ├── model/
│   └── resources/
├── A3-MedicineNameExtraction/
│   ├── README.md
│   ├── code/
│   ├── data/
│   ├── model/
│   └── resources/
└── A4-SentimentAnalysis/
    ├── README.md
    ├── code/
    ├── data/
    ├── logs/
    ├── report/
    └── Resources/
```

## Installation and Setup

Each project has its own dependencies and setup instructions. Please refer to the individual project READMEs for detailed installation guides.

### General Requirements

```bash
# Install common dependencies
pip install pandas numpy matplotlib seaborn

# For deep learning projects
pip install torch torchvision torchaudio

# For NLP tasks
pip install nltk gensim scikit-learn
```

## Academic Integrity

These projects were completed as part of the NLP course curriculum at Sharif University of Technology. The code and implementations are original work by the author, following academic guidelines and best practices.

## License

This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Taha Majlesi**

- Student ID: [Your Student ID]
- Email: [Your Email]
- GitHub: [Your GitHub Profile]

## Acknowledgments

# NLP Assignments — Sharif University (Spring 1402)

Clean, learning-focused presentation of course assignments and supporting code. This repo bundles four practical projects that walk from classic NLP preprocessing to feature-based models and deep learning for text.

Quick links

- Learning guide: `LEARNING_GUIDE.md`
- Requirements: `requirements.txt`
- Assignments: `A1-Persian-Poetry-Comments-on-Ganjoor/`, `A2-HeKasreh/`, `A3-MedicineNameExtraction/`, `A4-SentimentAnalysis/`

What this repository is for

This collection is designed for students who want a compact, hands-on path to learn modern NLP techniques with Persian- and English-language examples. Each assignment includes code, data samples or pointers, and a report/notebook describing experiments.

Who this is for

- Students or engineers new to NLP who want guided, project-based learning
- People interested in Persian NLP (datasets, Hazm-based preprocessing, morphological challenges)
- Practitioners who want runnable examples of embedding-based extraction and LSTM/PyTorch classification

How to use this repo (fast path)

1. Clone the repo and open it in your editor/IDE.
2. Create a Python environment and install dependencies from `requirements.txt` (recommended).
3. Read `LEARNING_GUIDE.md` for a suggested learning order and exercises.
4. Open each assignment's `README.md` and the `code/` folder. Notebooks are in the `report/` or `code/` directories for interactive exploration.

Project summaries (short)

- A1 — Persian Poetry Comments on Ganjoor: web-scraping + Persian preprocessing (Hazm), exploratory analysis, frequency/popularity analysis. See `A1-Persian-Poetry-Comments-on-Ganjoor/README.md`.
- A2 — HeKasreh: POS tagging for Persian with feature engineering and ML models; includes a trained `postagger.model`. See `A2-HeKasreh/README.md`.
- A3 — Medicine Name Extraction: cross-lingual name/entity extraction using embeddings and alignment (Gensim). Includes datasets and pre-trained embedding files. See `A3-MedicineNameExtraction/README.md`.
- A4 — Sentiment Analysis: LSTM-based sentiment classifier implemented in PyTorch; notebooks and training logs available. See `A4-SentimentAnalysis/README.md`.

Structure

```
nlp-assignments-spring-2023/
├─ README.md                 # This file (cleaned)
├─ LEARNING_GUIDE.md         # Suggested learning path and exercises (new)
├─ requirements.txt          # Consolidated dependencies (new)
├─ A1-Persian-Poetry-Comments-on-Ganjoor/
├─ A2-HeKasreh/
├─ A3-MedicineNameExtraction/
└─ A4-SentimentAnalysis/
```

Environment setup (recommended)

1. Create a virtual environment (venv or conda).

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

Notes on per-project extras

- Some assignments rely on large model or data files (word2vec/skip-gram models, datasets). These are included where feasible; otherwise the project README will point to download instructions.
- For Persian preprocessing we use `hazm` (see per-project README for version notes).

Contributing and usage

If you'd like this repo reorganized further (for example: turn each assignment into a small package, provide Docker/Conda files, or add unit tests), open an issue describing the desired output and I can prepare a follow-up pull request.

License

This repository is published under the MIT License — see `LICENSE`.

Author

Taha Majlesi

Acknowledgements

Thanks to the course instructors, TAs, and open-source library authors.
