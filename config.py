# config.py
DATA_DIR = "data"
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma2:2b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]