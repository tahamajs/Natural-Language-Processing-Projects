import unittest
import os
from model.bigram import BigramModel
from model.tokenizer import tokenize


class TestTopNext(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_dir, "data", "corpus.txt")
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenize(text)
        self.model = BigramModel()
        self.model.train(tokens)

    def test_top_next_the(self):
        top = self.model.top_next("the", n=2)
        self.assertEqual(top[0], "cat")
        self.assertIn(top[1], {"dog", "bird", "mat"})

    def test_top_next_i(self):
        top = self.model.top_next("i", n=1)
        self.assertEqual(top[0], "like")


if __name__ == "__main__":
    unittest.main()

