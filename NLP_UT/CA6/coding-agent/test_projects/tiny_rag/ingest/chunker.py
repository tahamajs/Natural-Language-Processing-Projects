class TextChunker:
    def __init__(self, chunk_size=100, overlap=0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        """
        Splits text into fixed-size chunks by words.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (self.chunk_size - self.overlap)
            
        return chunks