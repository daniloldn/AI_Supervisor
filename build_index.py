from src.loaders import load_pdf
from src.chunking import chunk_docs


def main():
    documents = load_pdf("data/raw")
    chunks = chunk_docs(documents, chunk_size=800, overlap=150)

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} chunks.\n")

    print("First 5 chunks:\n")
    for chunk in chunks[:5]:
        print("=" * 60)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk length: {len(chunk['text'])}")
        print("Preview:")
        print(chunk["text"][:300])
        print("\n")


if __name__ == "__main__":
    main()