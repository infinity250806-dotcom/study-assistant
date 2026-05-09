from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def get_embedding_model():
    """Load and return the HuggingFace embedding model."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


def create_vector_store(chunks, embedding_model):
    """Create a Chroma vector store from text chunks."""
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model
    )
    return vector_store