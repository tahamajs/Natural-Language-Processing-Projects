class Generator:
    def __init__(self, model):
        self.model = model
        import random

        self._random = random

    def generate(self, start_word, max_length=10):
        """
        Generates text starting with start_word.
        """
        current_word = start_word
        output = [current_word]

        for _ in range(max_length - 1):
            if current_word not in self.model.bigram_counts:
                break

            candidates = self.model.bigram_counts[current_word]
            if not candidates:
                break
            # Sample next word according to observed frequencies (weighted sampling)
            words = []
            weights = []
            for w, cnt in candidates.items():
                words.append(w)
                weights.append(cnt)

            # Use random.choices for weighted sampling
            next_word = self._random.choices(words, weights=weights, k=1)[0]

            output.append(next_word)
            current_word = next_word

        return " ".join(output)
