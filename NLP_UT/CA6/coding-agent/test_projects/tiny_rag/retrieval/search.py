def calculate_cosine_similarity(vec_a, vec_b):
    """
    Calculates cosine similarity between two vectors.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of same length")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    
    return dot_product

class VectorSearch:
    def __init__(self, database, vectorizer):
        self.db = database
        self.vectorizer = vectorizer

    def search(self, query, top_k=2):
        query_vector = self.vectorizer.embed(query)
        scores = []

        for doc_id, doc_vector in self.db.get_all_vectors().items():
            score = calculate_cosine_similarity(query_vector, doc_vector)
            scores.append((doc_id, score))
            
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]