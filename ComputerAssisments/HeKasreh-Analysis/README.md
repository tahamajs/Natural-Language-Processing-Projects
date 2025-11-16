# HeKasreh - Persian POS Tagging Toolkit

## Overview

This project implements a Part-of-Speech (POS) tagging system for the Persian language, focusing on the "He Kasreh" (ه کسره) correction and tagging. The system uses machine learning techniques to automatically assign POS tags to Persian words, which is essential for various NLP tasks such as syntactic parsing, information extraction, and text analysis.

The project includes data preprocessing, feature engineering, model training using various algorithms, and evaluation of the POS tagger's performance.

## Dataset

The dataset consists of Persian text corpora annotated with POS tags. The data is split into training and testing sets for model development and evaluation.

### Data Structure

- Training data: Annotated Persian sentences with POS tags
- Test data: Held-out dataset for evaluation
- Features: Word-level features including morphological, contextual, and lexical information

## Dependencies

- Python 3.x
- NLTK: For NLP preprocessing and evaluation
- scikit-learn: For machine learning algorithms
- pandas: For data manipulation
- numpy: For numerical operations
- pickle: For model serialization

## Installation

1. Install required packages:

```bash
pip install nltk scikit-learn pandas numpy
```

2. Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Usage

### Data Preprocessing

1. Load and preprocess the Persian text data:

```python
import pandas as pd
from nltk.tokenize import word_tokenize

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Tokenize sentences
df['tokens'] = df['text'].apply(word_tokenize)
```

2. Feature extraction for POS tagging:

```python
def extract_features(sentence, index):
    word = sentence[index]
    features = {
        'word': word,
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
        'is_all_lower': word.lower() == word,
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
    }
    return features
```

### Model Training

Train a POS tagger using various algorithms:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Prepare features and labels
X = [extract_features(sentence, i) for sentence in sentences for i in range(len(sentence))]
y = [tag for sentence_tags in tagged_sentences for tag in sentence_tags]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
with open('postagger.model', 'wb') as f:
    pickle.dump(model, f)
```

### Model Evaluation

Evaluate the trained model's performance:

```python
from sklearn.metrics import classification_report

# Load model
with open('postagger.model', 'rb') as f:
    model = pickle.load(f)

# Predict on test set
y_pred = model.predict(X_test)

# Print evaluation metrics
print(classification_report(y_test, y_pred))
```

### POS Tagging

Use the trained model to tag new Persian text:

```python
def pos_tag(sentence, model):
    features = [extract_features(sentence, i) for i in range(len(sentence))]
    tags = model.predict(features)
    return list(zip(sentence, tags))

# Example usage
sentence = ["این", "یک", "جمله", "تست", "است"]
tagged = pos_tag(sentence, model)
print(tagged)
```

## Model Details

The `postagger.model` file contains a trained machine learning model for Persian POS tagging. The model uses features such as:

- Word morphology (prefixes, suffixes)
- Capitalization patterns
- Contextual information (previous/next words)
- Word position in sentence

### Supported POS Tags

The system supports standard Persian POS tags including:

- Noun (N)
- Verb (V)
- Adjective (ADJ)
- Adverb (ADV)
- Preposition (P)
- Conjunction (CONJ)
- And others...

## Results

### Performance Metrics

- Accuracy: [X]% on test set
- Precision/Recall/F1-score for each POS tag
- Confusion matrix analysis

### Analysis

- The model performs well on common words but may struggle with rare or ambiguous terms
- Contextual features significantly improve tagging accuracy
- "He Kasreh" specific corrections improve handling of Persian morphological variations

## Project Structure

```
A2-HeKasreh/
├── README.md
├── code/
│   └── report.ipynb          # Jupyter notebook with implementation and analysis
├── data/
│   └── __init__.py
├── model/
│   └── postagger.model       # Trained POS tagging model
├── resources/
│   └── HW2_NLP_Spring1402.pdf # Assignment specification
└── src/
    └── __init__.py
```

## Features

- Persian text preprocessing and normalization
- Feature engineering for POS tagging
- Multiple machine learning algorithms support
- Model serialization and loading
- Comprehensive evaluation metrics
- "He Kasreh" morphological correction

## Future Improvements

- Incorporate deep learning models (LSTM, BERT)
- Add support for more POS tags
- Improve handling of out-of-vocabulary words
- Implement ensemble methods

## Contributors

- [Your Name]
- Course: Natural Language Processing, Sharif University of Technology
- Term: Spring 1402

## License

This project is licensed under the MIT License - see the LICENSE file for details.
