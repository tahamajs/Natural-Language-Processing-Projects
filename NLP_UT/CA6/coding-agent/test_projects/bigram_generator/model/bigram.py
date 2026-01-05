from collections import defaultdict, Counter
import math


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
            w2 = tokens[i + 1]

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
        # Laplace (add-one) smoothing to avoid zero probabilities
        V = max(1, len(self.vocab))
        count_w1 = self.unigram_counts.get(w1, 0)

        # If w1 unseen, return uniform probability over vocabulary
        if count_w1 == 0:
            return 1.0 / V

        count_bigram = self.bigram_counts[w1].get(w2, 0)
        return (count_bigram + 1) / (count_w1 + V)

    def score_sentence(self, tokens):
        """
        Calculates the total probability of a sequence of words.
        """
        # Use log-probabilities to avoid floating point underflow for long sequences.
        log_prob = 0.0
        for i in range(len(tokens) - 1):
            w1 = tokens[i]
            w2 = tokens[i + 1]
            p = self.get_probability(w1, w2)
            # guard against numerical issues
            p = max(p, 1e-300)
            log_prob += math.log(p)

        return log_prob

    def top_next(self, w1, n=3):
        """Return the top-n most likely next words after w1.

        Uses the model probabilities (with smoothing) to rank candidates.
        """
        # Consider full vocabulary
        candidates = list(self.vocab)
        # Compute probabilities for each candidate
        probs = []
        for w2 in candidates:
            probs.append((w2, self.get_probability(w1, w2)))

        # Sort by probability descending, then by word for deterministic order
        probs.sort(key=lambda x: (-x[1], x[0]))
        return [w for w, _ in probs[:n]]

    def top_k_next_words(self, w1, k):
        """
        Returns the top-k next words by probability given a word w1.
        """
        if w1 not in self.bigram_counts:
            return []
        next_words = self.bigram_counts[w1]
        sorted_next_words = sorted(
            next_words.items(), key=lambda item: item[1], reverse=True
        )
        return [word for word, count in sorted_next_words[:k]]
