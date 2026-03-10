import pickle
from pathlib import Path

def save_chunks_metadata(chunks, filepath="vectorstore/chunks_metadata.pkl"):
    Path("vectorstore").mkdir(exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(chunks, f)