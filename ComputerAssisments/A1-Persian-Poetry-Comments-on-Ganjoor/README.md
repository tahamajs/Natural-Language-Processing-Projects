# Persian Poetry Comments on Ganjoor - NLP Analysis

## Overview

This project focuses on the analysis of user comments collected from the Ganjoor website, a popular online repository of Persian poetry. The dataset includes real-world textual data in Persian from comments on poems by renowned poets such as Omar Khayyam, Hafez, Saadi, Ferdousi, Nezami, and Moulavi.

The project applies various Natural Language Processing (NLP) techniques including text cleaning, tokenization, normalization, stopword removal, and frequency analysis to extract insights from the comments.

## Dataset

The dataset is extracted from the Ganjoor website (https://ganjoor.net/) using web scraping techniques with the Scrapy framework. Comments are collected from the first 300 poems of each selected poet.

### Data Structure

Each comment entry in the JSONLines format contains:

- `comment_id`: Unique identifier for the comment
- `comment_author`: Name of the commenter (may be empty)
- `comment_text`: Array of strings containing the comment content
- `poet`: Name of the poet associated with the poem
- `poem_no`: Sequential identifier for the poem
- `poem_path`: URL path of the poem page
- `date_raw`: Raw date and time of the comment in Persian

### Sample Data

```json
{
  "comment_id": "comment-43263",
  "comment_author": " ",
  "comment_text": [
    "با درود خدمت اساتید",
    " \"آمیزاده نگه دار که مصحف ببرد\" به چه معناست؟ سپاسگزارم"
  ],
  "poet": "saadi",
  "poem_no": "1",
  "poem_path": "/saadi/nasr/sh6",
  "date_raw": "در ‫۳ سال و ۲ ماه قبل، دوشنبه ۱۱ آذر ۱۳۹۸، ساعت ۱۱:۴۷"
}
```

## Dependencies

- scrapy: For web crawling and data extraction
- crochet: For integrating asynchronous Scrapy with synchronous code
- jsonlines: For handling JSONLines format
- hazm: Persian NLP toolkit for normalization, tokenization, etc.
- dadmatools: Additional Persian text processing tools
- pandas: For data manipulation and analysis
- numpy: For numerical operations
- nltk: For frequency distribution analysis

## Installation

1. Install Python dependencies:

```bash
pip install scrapy crochet jsonlines hazm dadmatools pandas numpy nltk
```

## Usage

### Data Extraction

Run the Scrapy spider to extract comments from Ganjoor:

```python
from scrapy.crawler import CrawlerRunner
import crochet

@crochet.wait_for(240)
def run_spider():
    crawler = CrawlerRunner()
    d = crawler.crawl(GanjoorCommentsSpider)
    return d

run_spider()
```

This will generate `ganjoor.jsonlines` with the extracted comments.

### Data Processing

1. Load the data:

```python
import pandas as pd
import jsonlines

df = pd.read_json('ganjoor.jsonlines', lines=True)
```

2. Normalize text using Hazm:

```python
from hazm import Normalizer

normalizer = Normalizer(
    remove_extra_spaces=True,
    persian_style=True,
    persian_numbers=True,
    remove_diacritics=True,
    affix_spacing=True,
    punctuation_spacing=True
)

df["normalized_comment"] = df["comment_text"].apply(lambda comment: normalizer.normalize('\r\n'.join(comment)))
```

3. Tokenize and remove stopwords:

```python
from hazm import word_tokenize, sent_tokenize

# Sentence tokenization
df["sentences"] = df["normalized_comment"].apply(sent_tokenize)

# Word tokenization
df["tokens"] = df["sentences"].apply(lambda sentences: word_tokenize(''.join(sentences)))

# Remove stopwords
stop_words = hazm.stopwords_list() + [',', '.', ':', '«', '»', '(', ')', '،', '-', '?', '؟', '-']
df["filtered_tokens"] = df["tokens"].apply(lambda words: [w for w in words if w not in stop_words])
```

### Analysis

#### Comment Distribution by Poet

Group comments by poet and analyze distribution:

```python
data_per_poet = df.groupby(by=['poet'])
pages_per_poet = data_per_poet.count()['comment_id']
```

#### Word Frequency Analysis

Perform frequency analysis on filtered tokens:

```python
from nltk import FreqDist

freq_dist = {}
for poet, group in data_per_poet:
    all_tokens = [token for tokens in group['filtered_tokens'] for token in tokens]
    freq_dist[poet] = FreqDist(all_tokens).most_common(25)
```

#### Poet Reference Analysis

Analyze cross-references between poets in comments to identify similarities and comparisons.

## Results

### Comment Statistics

- Total comments extracted: [Number from dataset]
- Average comments per poem page by poet:
  - Saadi: [value]
  - Hafez: [value]
  - Khayyam: [value]
  - And others...

### Top Words by Poet

Frequency analysis reveals common themes and vocabulary used in comments for each poet.

### Insights

- Poets with higher comment volumes may indicate greater popularity or controversial content
- Cross-references between poets suggest thematic or stylistic similarities
- Common words highlight discussion topics like interpretation, historical context, and literary analysis

## Project Structure

```
A1-Persian-Poetry-Comments-on-Ganjoor/
├── README.md
├── code/
│   ├── ganjoor.jsonlines    # Extracted dataset
│   └── Report.ipynb         # Jupyter notebook with analysis
├── report/
│   └── CA1 ... .md          # Detailed report
└── assets/
    └── images/
        └── besm.png
```

## Contributors

- [Your Name]
- Course: Natural Language Processing, Sharif University of Technology
- Term: 4012

## License

This project is licensed under the MIT License - see the LICENSE file for details.
