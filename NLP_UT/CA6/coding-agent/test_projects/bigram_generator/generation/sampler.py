class Generator:
    def __init__(self, model):
        self.model = model

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

            next_word = candidates.most_common(1)[0][0]
            
            output.append(next_word)
            current_word = next_word
            
        return " ".join(output)