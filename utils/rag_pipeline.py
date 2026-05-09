from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    unique_chunks = list(dict.fromkeys(chunks))
    return unique_chunks


def extract_best_sentences(question, docs, embedding_model, top_n=5):

    # Split all chunks into sentences
    sentences = []
    for doc in docs:
        sentences += [s.strip() for s in doc.page_content.split(". ") if s.strip()]

    if not sentences:
        return "No relevant content found."

    # Embed question and sentences
    q_embedding = embedding_model.embed_query(question)
    s_embeddings = embedding_model.embed_documents(sentences)

    # Score each sentence against the question
    scores = cosine_similarity([q_embedding], s_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]

    # Return sentences in original order for readability
    ordered_indices = sorted(top_indices)
    answer = ". ".join([sentences[i] for i in ordered_indices])

    if not answer.endswith("."):
        answer += "."

    return answer