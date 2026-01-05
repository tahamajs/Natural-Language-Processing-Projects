from collections import defaultdict, Counter

class BigramModel:
    def __init__(self):
        # Stores counts of (word_i)
        self.unigram_counts = Counter()
        # Stores counts of (word_i, word_i+1)
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()

    def train(self, tokens):
        """Builds counts from a list of tokens."""
        for i in range(len(tokens) - 1):
            w1 = tokens[i]
            w2 = tokens[i+1]
            
            self.unigram_counts[w1] += 1
            self.bigram_counts[w1][w2] += 1
            self.vocab.add(w1)
            self.vocab.add(w2)
            
        if tokens:
            self.unigram_counts[tokens[-1]] += 1
            self.vocab.add(tokens[-1])

    def get_probability(self, w1, w2):
        """
        Calculates P(w2 | w1) = count(w1, w2) / count(w1)
        """
        count_w1 = self.unigram_counts[w1]
        if count_w1 == 0:
            return 0.0
            
        count_bigram = self.bigram_counts[w1][w2]
        return count_bigram / count_w1

    def score_sentence(self, tokens):
        """
        Calculates the total probability of a sequence of words.
        """
        prob = 1.0
        for i in range(len(tokens) - 1):
            w1 = tokens[i]
            w2 = tokens[i+1]
            p = self.get_probability(w1, w2)
            prob *= p
            
        return prob