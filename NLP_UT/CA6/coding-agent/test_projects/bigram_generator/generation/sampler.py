class Generator:
    def __init__(self, model):
        self.model = model
        import random
        import math

        self._random = random
        self._math = math

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

    def generate_with_temperature(self, start_word, max_length=10, temperature=1.0):
        """
        Generates text with temperature sampling for diversity.

        Args:
            start_word (str): The starting word for generation.
            max_length (int): Maximum length of generated sequence.
            temperature (float): Temperature for sampling. Lower values (<1.0) make output more deterministic,
                                higher values (>1.0) make it more diverse/random.

        Returns:
            str: Generated text sequence.
        """
        current_word = start_word
        output = [current_word]

        for _ in range(max_length - 1):
            if current_word not in self.model.bigram_counts:
                break

            candidates = self.model.bigram_counts[current_word]
            if not candidates:
                break

            # Calculate probabilities
            total_count = sum(candidates.values())
            probs = {}
            for w, cnt in candidates.items():
                probs[w] = cnt / total_count

            # Apply temperature
            if temperature != 1.0:
                # Convert to logits (log probs), apply temperature, convert back to probs
                logits = {w: self._math.log(p) / temperature for w, p in probs.items()}
                # Normalize to probabilities
                max_logit = max(logits.values())
                exp_logits = {w: self._math.exp(logit - max_logit) for w, logit in logits.items()}
                total_exp = sum(exp_logits.values())
                probs = {w: exp / total_exp for w, exp in exp_logits.items()}

            # Sample from the distribution
            words = list(probs.keys())
            probabilities = [probs[w] for w in words]
            next_word = self._random.choices(words, weights=probabilities, k=1)[0]

            output.append(next_word)
            current_word = next_word

        return " ".join(output)
