import streamlit as st
from utils.pdf_loader import load_pdf
from utils.rag_pipeline import split_text, generate_answer
from utils.embeddings import create_vector_store, get_embedding_model

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
}

[data-testid="stAppViewContainer"] {
    background: 
        radial-gradient(ellipse 80% 50% at 20% 0%, rgba(99, 57, 255, 0.15) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(0, 210, 190, 0.1) 0%, transparent 60%),
        #0a0a0f !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

.block-container {
    max-width: 760px !important;
    padding: 3rem 2rem !important;
}

/* Title */
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: rgba(255,255,255,0.35);
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(167, 139, 250, 0.3) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(167, 139, 250, 0.6) !important;
}

[data-testid="stFileUploader"] label {
    color: rgba(255,255,255,0.5) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}

/* Success box */
[data-testid="stAlert"] {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.12), rgba(52, 211, 153, 0.05)) !important;
    border: 1px solid rgba(52, 211, 153, 0.3) !important;
    border-radius: 12px !important;
    color: #34d399 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Text input */
[data-testid="stTextInput"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.4) !important;
}

[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.9rem 1.2rem !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: rgba(167, 139, 250, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.1) !important;
    outline: none !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: rgba(255,255,255,0.2) !important;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: #a78bfa !important;
}

/* Answer card */
.answer-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}

.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #a78bfa, #34d399, #60a5fa);
    border-radius: 20px 20px 0 0;
}

.answer-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1rem;
}

.answer-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    line-height: 1.8;
    color: rgba(255,255,255,0.85);
    font-weight: 300;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 2rem 0;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">Study Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by RAG · Ask anything from your notes</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = load_pdf(uploaded_file)
        chunks = split_text(text)
        embedding_model = get_embedding_model()
        vector_store = create_vector_store(chunks, embedding_model)

    st.success("✓ PDF processed successfully")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    question = st.text_input("Ask a Question", placeholder="e.g. What is Design Thinking?")

    if question:
        with st.spinner("Thinking..."):
            docs = vector_store.similarity_search(question, k=3)
            answer = generate_answer(question, docs)

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">Answer</div>
            <div class="answer-text">{answer}</div>
        </div>
        """, unsafe_allow_html=True)