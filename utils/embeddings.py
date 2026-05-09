from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model
    )

    return vector_store