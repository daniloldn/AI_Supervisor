def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a grounded prompt using retrieved context.
    """
    context_blocks = []

    for chunk in retrieved_chunks:
        block = f"[Source: {chunk['source']}]\n{chunk['text']}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are an academic supervisor helping a university student understand course material deeply.

Use ONLY the provided context to answer the question.
If the answer cannot be found in the context, say:
"I don’t know based on the provided materials. I can only answer question based on pre-indexed material"

Be rigorous but supportive.
- Explain clearly
- Use step-by-step reasoning when helpful
- Point out important assumptions
- Reference the source documents when relevant
- Point out what could come up in the exam 

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt.strip()