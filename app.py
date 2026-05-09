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

        # Extract text from PDF
        text = load_pdf(uploaded_file)

        # Split into chunks
        chunks = split_text(text)

        # Create vector database
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

        # Get retrieved chunk
        answer = " ".join([doc.page_content for doc in docs])

        # Clean extra spaces/newlines
        answer = re.sub(r"\s+", " ", answer)

        # Split into sentences
        sentences = answer.split(". ")

        # Remove duplicate sentences
        unique_sentences = []
        seen = set()

        for sentence in sentences:

            clean_sentence = sentence.strip()

            if clean_sentence and clean_sentence not in seen:
                unique_sentences.append(clean_sentence)
                seen.add(clean_sentence)

        # Join cleaned sentences
        final_answer = ". ".join(unique_sentences)

        # Add final period
        if not final_answer.endswith("."):
            final_answer += "."

        st.write(final_answer)