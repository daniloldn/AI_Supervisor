import faiss
import numpy as np
from pathlib import Path
import pickle


def build_faiss_index(embedded_chunks: list[dict], index_path="vectorstore/faiss.index"):
    """
    Build and save FAISS index from embedded chunks.
    """
    Path("vectorstore").mkdir(exist_ok=True)

    embeddings = np.array([chunk["embedding"] for chunk in embedded_chunks]).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    print(f"FAISS index saved to {index_path}")


def save_metadata(embedded_chunks, metadata_path="vectorstore/metadata.pkl"):
    """
    Save chunk metadata without embeddings.
    """
    clean_chunks = []

    for chunk in embedded_chunks:
        clean_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"]
        })

    with open(metadata_path, "wb") as f:
        pickle.dump(clean_chunks, f)

    print(f"Metadata saved to {metadata_path}")


def load_faiss_index(index_path="vectorstore/faiss.index"):
    """
    Load a saved FAISS index from disk.
    """
    return faiss.read_index(index_path)


def load_metadata(metadata_path="vectorstore/metadata.pkl"):
    """
    Load saved chunk metadata from disk.
    """
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return metadata


def search_index(query_embedding: list[float], index, metadata, k: int = 4) -> list[dict]:
    """
    Search the FAISS index and return top-k matching chunks.
    """
    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vector, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue

        chunk = metadata[idx].copy()
        chunk["distance"] = float(dist)
        results.append(chunk)

    return results