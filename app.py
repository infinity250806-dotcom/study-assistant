import streamlit as st
from utils.pdf_loader import load_pdf
from utils.rag_pipeline import split_text, extract_best_sentences
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
        with st.spinner("Searching Notes..."):
            # Retrieve top 5 relevant chunks
            docs = vector_store.similarity_search(question, k=5)

            # Extract best sentences — top_n=8 to capture full lists
            answer = extract_best_sentences(question, docs, embedding_model, top_n=8)

        st.subheader("Answer")
        st.write(answer)