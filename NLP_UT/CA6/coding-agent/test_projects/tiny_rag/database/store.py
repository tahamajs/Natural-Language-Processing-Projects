class InMemoryStore:
    def __init__(self):
        self.vectors = {} # doc_id -> vector
        self.documents = {} # doc_id -> text content

    def add_document(self, doc_id, text, vector):
        self.documents[doc_id] = text
        self.vectors[doc_id] = vector

    def get_document(self, doc_id):
        return self.documents.get(doc_id)

    def get_all_vectors(self):
        return self.vectors