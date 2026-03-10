from pathlib import Path
from src.loaders import load_pdf
from src.chunking import chunk_docs
from src.embedding import embed_chunks
from src.retrieval import build_faiss_index, save_metadata


def main():
    documents = load_pdf("data/raw")
    chunks = chunk_docs(documents, chunk_size=800, overlap=150)

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} chunks.")

    embedded_chunks = embed_chunks(chunks)

    print("Building FAISS index...")
    build_faiss_index(embedded_chunks)

    print("Saving metadata...")
    save_metadata(embedded_chunks)

    print("Done.")


if __name__ == "__main__":
    main()