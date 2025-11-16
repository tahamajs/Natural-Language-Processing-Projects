# Medicine Name Extraction - Cross-Lingual NLP

## Overview

This project implements a system for extracting medicine names from textual data using cross-lingual word embeddings and machine learning techniques. The system handles both English and Persian text, leveraging aligned word embeddings to identify and extract medication names from various sources.

The project combines natural language processing, word embeddings, and similarity-based approaches to accurately identify medicine names in multilingual contexts.

## Dataset

The project uses multiple datasets for training and evaluation:

### Primary Datasets

1. **English Medicine Dataset**: Collection of English medical texts with annotated medicine names
2. **Persian Medicine Dataset**: Persian medical texts and drug information
3. **Iran FDA Dataset**: Official drug database from Iranian Food and Drug Administration
4. **WHO ATC Index**: World Health Organization Anatomical Therapeutic Chemical classification system

### Data Structure

- `english_df.csv`: English medicine names and descriptions
- `persian_df.csv`: Persian medicine names and descriptions
- `medicine_ds_merged.csv`: Combined dataset
- `Iran_FDA_1400_Dataset.csv`: Iranian drug database
- `WHO_ATC_Index.csv`: WHO classification data

## Dependencies

- Python 3.x
- gensim: For word embeddings and similarity calculations
- scikit-learn: For machine learning algorithms
- pandas: For data manipulation
- numpy: For numerical operations
- matplotlib/seaborn: For visualization (optional)

## Installation

1. Install required packages:

```bash
pip install gensim scikit-learn pandas numpy matplotlib seaborn
```

## Usage

### Data Preparation

1. Load and preprocess the datasets:

```python
import pandas as pd

# Load datasets
english_df = pd.read_csv('data/english_df.csv')
persian_df = pd.read_csv('data/persian_df.csv')
iran_fda = pd.read_csv('data/Iran_FDA_1400_Dataset.csv')
who_atc = pd.read_csv('data/WHO_ATC_Index.csv')

# Merge datasets
merged_df = pd.concat([english_df, persian_df], ignore_index=True)
```

### Word Embeddings Training

Train Skip-gram models for both languages:

```python
from gensim.models import Word2Vec
import pickle

# Train English embeddings
english_sentences = [text.split() for text in english_df['description']]
english_model = Word2Vec(english_sentences, vector_size=300, window=5, min_count=1, sg=1)
english_model.save('model/en_skipgram.model')

# Train Persian embeddings
persian_sentences = [text.split() for text in persian_df['description']]
persian_model = Word2Vec(persian_sentences, vector_size=300, window=5, min_count=1, sg=1)
persian_model.save('model/fa_skipgram.model')
```

### Embedding Alignment

Align English and Persian embeddings using mapping techniques:

```python
# Load pre-trained aligned embeddings
english_aligned = Word2Vec.load('model/english_aligned.model')
persian_aligned = Word2Vec.load('model/persian_aligned.model')
```

### Medicine Name Extraction

Implement extraction using similarity-based approaches:

```python
def extract_medicine_names(text, model, medicine_list, threshold=0.8):
    tokens = text.split()
    extracted_names = []

    for token in tokens:
        if token in model.wv:
            similarities = [model.wv.similarity(token, med) for med in medicine_list if med in model.wv]
            if similarities and max(similarities) > threshold:
                extracted_names.append(token)

    return extracted_names

# Example usage
text = "Patient was prescribed aspirin and metformin"
medicine_list = ["aspirin", "metformin", "ibuprofen"]
extracted = extract_medicine_names(text, english_model, medicine_list)
print(extracted)
```

### Cross-Lingual Extraction

Use aligned embeddings for cross-lingual medicine name recognition:

```python
def cross_lingual_extraction(text, source_model, target_model, medicine_dict, threshold=0.7):
    # Translate medicine names using aligned embeddings
    translated_meds = {}
    for eng_med, pers_med in medicine_dict.items():
        if eng_med in source_model.wv and pers_med in target_model.wv:
            # Find similar words in target language
            similar_words = target_model.wv.similar_by_word(pers_med, topn=5)
            translated_meds[eng_med] = [word for word, _ in similar_words]

    # Extract from text
    extracted = []
    tokens = text.split()
    for token in tokens:
        for eng_med, translations in translated_meds.items():
            if token in translations:
                extracted.append((token, eng_med))
                break

    return extracted
```

## Model Files

- `en_skipgram.model`: English Skip-gram word embeddings
- `fa_skipgram.model`: Persian Skip-gram word embeddings
- `english_aligned.model`: Aligned English embeddings
- `persian_aligned.model`: Aligned Persian embeddings
- `en_embeddings.txt`: English embedding vectors
- `fa_embeddings.txt`: Persian embedding vectors
- `en_mapped_embeddings.txt`: Mapped English embeddings
- `fa_mapped_embeddings.txt`: Mapped Persian embeddings

## Evaluation

### Test Queries

The `test_input_queries.csv` file contains test queries for evaluating the extraction system.

### Metrics

- Precision: Accuracy of extracted medicine names
- Recall: Coverage of actual medicine names
- F1-Score: Harmonic mean of precision and recall
- Cross-lingual accuracy: Performance across languages

## Results

### Performance

- English extraction accuracy: [X]%
- Persian extraction accuracy: [X]%
- Cross-lingual transfer accuracy: [X]%
- Top similar medicines identified correctly

### Analysis

- Word embeddings effectively capture semantic relationships between medicine names
- Alignment improves cross-lingual performance
- Skip-gram architecture outperforms CBOW for rare medicine names
- Contextual information enhances extraction accuracy

## Project Structure

```
A3-MedicineNameExtraction/
├── README.md
├── LICENSE
├── test_input_queries.csv
├── .vscode/
│   └── settings.json
├── code/
│   └── report.ipynb           # Implementation and analysis notebook
├── data/
│   ├── english_df.csv
│   ├── persian_df.csv
│   ├── medicine_ds_merged.csv
│   ├── Iran_FDA_1400_Dataset.csv
│   ├── WHO_ATC_Index.csv
│   └── orig/                  # Original unmodified datasets
├── model/
│   ├── en_embeddings.txt
│   ├── en_mapped_embeddings.txt
│   ├── en_skipgram.model
│   ├── english_aligned.model
│   ├── fa_embeddings.txt
│   ├── fa_mapped_embeddings.txt
│   ├── fa_skipgram.model
│   └── persian_aligned.model
└── resources/
    └── HW3.pdf                # Assignment specification
```

## Features

- Cross-lingual medicine name extraction
- Word embedding alignment for language transfer
- Similarity-based entity recognition
- Support for English and Persian text
- Integration with official drug databases (FDA, WHO)
- Comprehensive evaluation framework

## Applications

- Automated drug information extraction from medical texts
- Pharmacovigilance and adverse drug reaction monitoring
- Medical information retrieval systems
- Cross-lingual medical document analysis
- Drug name normalization and standardization

## Future Improvements

- Incorporate contextual embeddings (BERT, ELMo)
- Add support for more languages
- Implement sequence labeling approaches (CRF, BiLSTM)
- Improve handling of drug name variations and abbreviations
- Add confidence scores for extracted entities

## Contributors

- [Your Name]
- Course: Natural Language Processing, Sharif University of Technology
- Term: Spring 1402

## License

This project is licensed under the MIT License - see the LICENSE file for details.
