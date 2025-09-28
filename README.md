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

- Special thanks to the course instructor and teaching assistants
- Appreciation to the Sharif University of Technology for providing the academic environment
- Gratitude to the open-source community for the excellent libraries and tools used in these projects

---

_This repository showcases the practical application of NLP concepts learned throughout the course, demonstrating proficiency in text processing, machine learning, and deep learning techniques for real-world language understanding tasks._
