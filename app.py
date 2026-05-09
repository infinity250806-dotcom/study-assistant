import streamlit as st
from utils.pdf_loader import load_pdf
from utils.rag_pipeline import split_text, generate_answer
from utils.embeddings import create_vector_store, get_embedding_model

st.set_page_config(page_title="AI Study Assistant")
st.title("📚 AI Study Assistant")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = load_pdf(uploaded_file)
        chunks = split_text(text)
        embedding_model = get_embedding_model()
        vector_store = create_vector_store(chunks, embedding_model)

    st.success("PDF Processed Successfully")

    question = st.text_input("Ask a Question")

    if question:
        with st.spinner("Generating Answer..."):
            docs = vector_store.similarity_search(question, k=3)
            answer = generate_answer(question, docs)

        st.subheader("Answer")
        st.write(answer)