import unittest
import os
from model.bigram import BigramModel
from model.tokenizer import tokenize
from generation.sampler import Generator

class TestLanguageModel(unittest.TestCase):
    
    def setUp(self):
        self.model = BigramModel()
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_dir, 'data', 'corpus.txt')
        
        with open(data_path, 'r') as f:
            text = f.read()
            
        self.tokens = tokenize(text)
        self.model.train(self.tokens)

    def test_smoothing_unseen_bigrams(self):
        """
        Test that the model returns a small non-zero probability for unseen pairs
        instead of crashing or returning 0.0.
        """
        prob = self.model.get_probability("cat", "fly")
        
        self.assertGreater(prob, 0.0, "Zero Probability Error.")

    def test_underflow_long_sequence(self):
        """
        Test that scoring a long sequence doesn't vanish to 0.0.
        """
        long_seq = ["the", "cat"] * 50
        
        score = self.model.score_sentence(long_seq)
        
        self.assertNotEqual(score, 0.0, 
            "Floating Point Underflow: The probability vanished to 0.0.\n"
            "Switch to Log-Probabilities to handle long sequences.")
            
        if score > 0:
             self.fail("Use Log Probabilities for numerical stability.")

    def test_generation_diversity(self):
        """
        Test that generation is not deterministic.
        """
        gen = Generator(self.model)
        
        outputs = set()
        for _ in range(10):
            sent = gen.generate("i", max_length=5)
            outputs.add(sent)
            
        self.assertGreater(len(outputs), 1, 
            f"Greedy Sampling Detected: Generated 10 identical sentences:\n{list(outputs)[0]}\n")

if __name__ == "__main__":
    unittest.main()