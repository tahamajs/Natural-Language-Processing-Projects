import re
from collections import Counter

class SimpleVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def build_vocab(self, chunks):
        """
        Builds a vocabulary from all documents.
        """
        all_words = set()
        for chunk in chunks:
            words = self._tokenize(chunk)
            all_words.update(words)
        
        self.vocab = {word: i for i, word in enumerate(sorted(all_words))}
        # Build simple IDF statistics
        df = {}
        total_docs = len(chunks) if len(chunks) > 0 else 1
        for word in self.vocab:
            df[word] = 0
        for chunk in chunks:
            seen = set(self._tokenize(chunk))
            for w in seen:
                if w in df:
                    df[w] += 1

        import math
        # Use smoothed IDF to avoid zero weights for df=1 in tiny corpora
        self.idf = {w: math.log(total_docs / max(1, df[w])) + 1.0 for w in self.vocab}

    def embed(self, text):
        """
        Converts text to a vector based on word frequency.
        """
        words = self._tokenize(text)
        word_counts = Counter(words)
        
        vector = [0] * len(self.vocab)
        
        for word, count in word_counts.items():
            if word in self.vocab:
                index = self.vocab[word]

                vector[index] = count
        # Apply TF-IDF weighting
        total = sum(vector)
        tfidf = [0.0] * len(self.vocab)
        for word, idx in self.vocab.items():
            tf = vector[idx] / total if total > 0 else 0.0
            idf = self.idf.get(word, 0.0)
            tfidf[idx] = tf * idf

        # Normalize vector (L2) to make cosine meaningful
        norm = sum(x * x for x in tfidf) ** 0.5
        if norm > 0:
            tfidf = [x / norm for x in tfidf]

        return tfidf

    def _tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Simple stopword removal to avoid noisy tokens like 'the' dominating results
        stopwords = {"the", "a", "an", "is", "of", "and", "to", "in"}
        return [t for t in tokens if t not in stopwords]