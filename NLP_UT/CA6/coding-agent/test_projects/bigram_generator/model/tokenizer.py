import re

def tokenize(text):
    """Converts text to lowercase and tokenizes it."""
    text = text.lower()
    return re.findall(r'\b\w+\b', text)