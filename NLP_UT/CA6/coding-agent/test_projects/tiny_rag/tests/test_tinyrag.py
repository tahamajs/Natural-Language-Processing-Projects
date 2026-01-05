import unittest
import os
from ingest.loader import load_text_file
from ingest.chunker import TextChunker
from retrieval.embedding import SimpleVectorizer
from retrieval.search import VectorSearch, calculate_cosine_similarity
from database.store import InMemoryStore

class TestEndToEndPipeline(unittest.TestCase):
    """
    Integration tests using the full application stack
    """

    def setUp(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.data_path = os.path.join(base_dir, 'data', 'knowledge_base.txt')
        
        try:
            self.raw_text = load_text_file(self.data_path)
            self.lines = [line for line in self.raw_text.splitlines() if line.strip()]
        except Exception as e:
            self.fail(f"Pipeline crashed at Loader stage: {e}")

        self.db = InMemoryStore()

    def test_chunking_logic(self):
        """
        Tests the Chunker component in isolation using loaded data.
        """
        input_text = self.lines[0] 
        
        chunker = TextChunker(chunk_size=10, overlap=0)
        chunks = chunker.chunk_text(input_text)
                
        self.assertEqual(
            chunks[0],
            self.lines[0],
            f"Chunker Logic Failure.\n"
            f"Input: '{self.lines[0]}'\n"
            f"Bad Output: {chunks[0]}\n"
        )

    def test_search_quality(self):
        """
        Tests the full Retrieval flow (Vectorizer + Store + Search).
        """
        vectorizer = SimpleVectorizer()
        
        doc1 = self.lines[1]
        doc2 = self.lines[2]
        
        vectorizer.build_vocab([doc1, doc2])
        
        vec1 = vectorizer.embed(doc1)
        vec2 = vectorizer.embed(doc2)
        
        self.db.add_document("doc1", doc1, vec1)
        self.db.add_document("doc2", doc2, vec2)
        
        search_engine = VectorSearch(self.db, vectorizer)
        
        query = "The Intelligence"
        results = search_engine.search(query, top_k=2)
        
        top_doc_id = results[0][0]
        scores_dict = {doc_id: score for doc_id, score in results}
        
        self.assertNotEqual(
            top_doc_id, 
            "doc1",
            f"Search Quality Failure: The query '{query}' retrieved the Noise document first.\n"
            f"Noise Score: {scores_dict.get('doc1')}\n"
            f"Content Score: {scores_dict.get('doc2')}\n"
        )

    def test_similarity_score(self):
        """
        Tests the cosine similarity calculation.
        """
        vectorizer = SimpleVectorizer()
        
        doc1 = self.lines[0]
        
        vectorizer.build_vocab([doc1])
        
        vec1 = vectorizer.embed(doc1)

        sim_score = calculate_cosine_similarity(vec1, vec1)
        
        self.assertAlmostEqual(
            sim_score, 
            1.0, 
            places=2,
            msg=(
                f"Math Error in Scoring: A document's similarity to itself is {sim_score:.2f}, expected 1.0.\n"
            )
        )

if __name__ == "__main__":
    unittest.main()