import streamlit as st
import re

from utils.pdf_loader import load_pdf
from utils.rag_pipeline import split_text
from utils.embeddings import create_vector_store

st.set_page_config(page_title="AI Study Assistant")

st.title("📚 AI Study Assistant")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        text = load_pdf(uploaded_file)

        chunks = split_text(text)

        vector_store = create_vector_store(chunks)

    st.success("PDF Processed Successfully")

    question = st.text_input("Ask a Question")

    if question:

        with st.spinner("Searching Notes..."):

            docs = vector_store.similarity_search(
                question,
                k=1
            )

        st.subheader("Answer")

        clean_answer = docs[0].page_content

        clean_answer = re.sub(r"\s+", " ", clean_answer)

        st.write(clean_answer)