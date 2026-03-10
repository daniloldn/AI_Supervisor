from src.embedding import embed_text
from src.retrieval import load_faiss_index, load_metadata, search_index
from src.prompting import build_prompt
from src.llm import generate_answer


def main():
    index = load_faiss_index()
    metadata = load_metadata()

    query = "What is the difference between horizontal and vertical differenitation?"
    query_embedding = embed_text(query)

    retrieved = search_index(query_embedding, index, metadata, k=4)

    prompt = build_prompt(query, retrieved)

    print("Generating answer...\n")
    answer = generate_answer(prompt)

    print("=" * 70)
    print("ANSWER:\n")
    print(answer)


if __name__ == "__main__":
    main()