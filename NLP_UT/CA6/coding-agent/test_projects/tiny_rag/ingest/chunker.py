class TextChunker:
    def __init__(self, chunk_size=100, overlap=0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        """
        Splits text into fixed-size chunks by words.
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))

        return chunks