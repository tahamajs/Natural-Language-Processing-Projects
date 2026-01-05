import re
from collections import Counter

class SimpleVectorizer:
    def __init__(self):
        self.vocab = {}

    def build_vocab(self, chunks):
        """
        Builds a vocabulary from all documents.
        """
        all_words = set()
        for chunk in chunks:
            words = self._tokenize(chunk)
            all_words.update(words)
        
        self.vocab = {word: i for i, word in enumerate(sorted(all_words))}

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
                
        return vector

    def _tokenize(self, text):
        text = text.lower()
        return re.findall(r'\b\w+\b', text)