import streamlit as st

from src.embedding import embed_text
from src.retrieval import load_faiss_index, load_metadata, search_index
from src.prompting import build_prompt
from src.llm import generate_answer


st.set_page_config(page_title="AI Academic Supervisor")

st.title("AI Industrial Economics Lecturer")
st.write("A RAG system to learn more about Industrial Economics")


@st.cache_resource
def load_resources():
    index = load_faiss_index()
    metadata = load_metadata()
    return index, metadata


index, metadata = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your course materials...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            query_embedding = embed_text(question)
            retrieved_chunks = search_index(query_embedding, index, metadata, k=4)
            prompt = build_prompt(question, retrieved_chunks)
            answer = generate_answer(prompt)

        st.markdown(answer)

        with st.expander("Sources used"):
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.markdown(f"**{i}. {chunk['source']}**")
                st.caption(f"Chunk ID: {chunk['chunk_id']} | Distance: {chunk['distance']:.4f}")
                st.write(chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"])
                st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})