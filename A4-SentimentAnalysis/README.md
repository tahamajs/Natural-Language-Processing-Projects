# Comprehensive Sentiment Analysis on Text Data

## Overview

This project implements a comprehensive sentiment analysis system for textual data, focusing on movie reviews. The system uses deep learning techniques, specifically Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units, to classify the sentiment of text as positive or negative.

The project includes data preprocessing, model architecture design, training, evaluation, and deployment considerations for sentiment classification.

## Dataset

The dataset consists of movie reviews with sentiment labels:

### Data Files

- `movie_train.jsonl`: Training set with movie reviews and sentiment labels
- `movie_dev.jsonl`: Development/validation set
- `movie_test.jsonl`: Test set for final evaluation
- `movie.jsonl`: Complete dataset

### Data Structure

Each entry in JSONL format contains:

- `text`: The movie review text
- `label`: Sentiment label (0 for negative, 1 for positive)
- `id`: Unique identifier for the review

### Sample Data

```json
{"text": "This movie was excellent! The acting was superb and the plot kept me engaged throughout.", "label": 1, "id": "12345"}
{"text": "Terrible film. Waste of time and money. The story made no sense.", "label": 0, "id": "67890"}
```

## Dependencies

- Python 3.x
- PyTorch: Deep learning framework
- torchtext: Text processing utilities for PyTorch
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Evaluation metrics
- matplotlib: Visualization
- tensorboard: Experiment tracking

## Installation

1. Install PyTorch (adjust for your system):

```bash
pip install torch torchvision torchaudio
```

2. Install other dependencies:

```bash
pip install torchtext pandas numpy scikit-learn matplotlib tensorboard
```

## Usage

### Data Preprocessing

1. Load and preprocess the data:

```python
import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Load data
train_df = pd.read_json('data/movie_train.jsonl', lines=True)
dev_df = pd.read_json('data/movie_dev.jsonl', lines=True)
test_df = pd.read_json('data/movie_test.jsonl', lines=True)

# Tokenization
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_df['text']), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
```

2. Create data loaders:

```python
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(vocab(tokenizer(_text)), dtype=torch.int64)
        text_list.append(processed_text)
    return pad_sequence(text_list, padding_value=0.0), torch.tensor(label_list)

# Create datasets
train_dataset = list(zip(train_df['text'], train_df['label']))
dev_dataset = list(zip(dev_df['text'], dev_df['label']))

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)
```

### Model Architecture

Implement LSTM-based sentiment classifier:

```python
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

# Model parameters
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1
n_layers = 2
dropout = 0.5

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
```

### Training

Train the model with appropriate loss function and optimizer:

```python
import torch.optim as optim

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for texts, labels in iterator:
        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Run training
n_epochs = 10
for epoch in range(n_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.3f}')
```

### Evaluation

Evaluate model performance on validation and test sets:

```python
from sklearn.metrics import classification_report, accuracy_score

def evaluate(model, iterator):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for texts, labels in iterator:
            preds = torch.sigmoid(model(texts).squeeze(1)) > 0.5
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return predictions, true_labels

# Evaluate on dev set
dev_preds, dev_labels = evaluate(model, dev_loader)
print(f'Accuracy: {accuracy_score(dev_labels, dev_preds):.3f}')
print(classification_report(dev_labels, dev_preds))
```

### Model Saving and Loading

```python
# Save model
torch.save(model.state_dict(), 'sentiment_model.pth')

# Load model
model.load_state_dict(torch.load('sentiment_model.pth'))
model.eval()
```

## Model Details

### Architecture

- **Embedding Layer**: Converts words to dense vectors
- **LSTM Layers**: Captures sequential dependencies in text (2 layers with dropout)
- **Fully Connected Layer**: Binary classification output
- **Dropout**: Prevents overfitting

### Hyperparameters

- Embedding dimension: 100
- Hidden dimension: 256
- Number of LSTM layers: 2
- Dropout rate: 0.5
- Learning rate: 0.001 (Adam optimizer)

## Results

### Performance Metrics

- **Accuracy**: [X]% on test set
- **Precision**: [X]% for positive class
- **Recall**: [X]% for positive class
- **F1-Score**: [X]%

### Training Logs

TensorBoard logs are available in the `logs/` directory for monitoring training progress.

### Analysis

- LSTM effectively captures long-range dependencies in reviews
- Dropout and proper regularization prevent overfitting
- Model performs well on longer reviews with clear sentiment
- Challenges with neutral or mixed sentiment reviews

## Project Structure

```
A4-SentimentAnalysis/
├── README.md
├── LICENSE
├── .vscode/
│   └── settings.json
├── code/
│   └── report_rev1.ipynb      # Implementation notebook
├── data/
│   ├── movie_dev.jsonl
│   ├── movie_test.jsonl
│   ├── movie_train.jsonl
│   └── movie.jsonl
├── logs/
│   ├── events.out.tfevents... # TensorBoard logs
│   └── [timestamp]/
│       └── events.out.tfevents...
├── report/
│   └── report.html            # HTML report
└── Resources/
    └── NLP_Spring1401_HW4.pdf # Assignment specification
```

## Features

- LSTM-based sentiment classification
- Comprehensive data preprocessing pipeline
- TensorBoard integration for experiment tracking
- Evaluation metrics and confusion matrix analysis
- Model serialization for deployment
- Support for variable-length text inputs

## Applications

- Movie review sentiment analysis
- Product review classification
- Social media sentiment monitoring
- Customer feedback analysis
- Text classification for various domains

## Future Improvements

- Incorporate pre-trained embeddings (GloVe, Word2Vec)
- Use transformer-based models (BERT, RoBERTa)
- Add attention mechanisms
- Implement multi-class sentiment analysis
- Add model interpretability features

## Contributors

- [Your Name]
- Course: Natural Language Processing, Sharif University of Technology
- Term: Spring 1401

## License

This project is licensed under the MIT License - see the LICENSE file for details.
