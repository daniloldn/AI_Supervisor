from openai import OpenAI
from src.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str) -> list[float]:
    """
    Generate embedding vector for input text.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add embeddings to each chunk dictionary.
    """
    embedded_chunks = []

    for chunk in chunks:
        embedding = embed_text(chunk["text"])

        embedded_chunks.append(
            {
                **chunk,
                "embedding": embedding
            }
        )

    return embedded_chunks