from src.embedding import embed_text
from src.retrieval import load_faiss_index, load_metadata, search_index


def main():
    index = load_faiss_index()
    metadata = load_metadata()

    query = "What is a horizontal merger?"
    query_embedding = embed_text(query)

    results = search_index(query_embedding, index, metadata, k=4)

    print(f"Query: {query}\n")
    print(f"Retrieved {len(results)} chunks:\n")

    for i, result in enumerate(results, 1):
        print("=" * 60)
        print(f"Rank: {i}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Distance: {result['distance']:.4f}")
        print("Preview:")
        print(result["text"][:400])
        print("\n")


if __name__ == "__main__":
    main()