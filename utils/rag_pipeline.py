from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    # Remove empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Remove duplicates while preserving order
    unique_chunks = list(dict.fromkeys(chunks))

    return unique_chunks


def clean_and_split_sentences(text):
    """
    Properly split text into sentences without breaking numbered lists.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split on sentence-ending punctuation but NOT on numbered list patterns like "1. "
    # Pattern: split after ". " only if NOT preceded by a digit
    sentences = re.split(r'(?<![0-9])\. (?=[A-Z])', text)

    cleaned = []
    for s in sentences:
        s = s.strip()
        if s:
            # Add period back if missing
            if not s.endswith("."):
                s += "."
            cleaned.append(s)

    return cleaned


def remove_duplicates(sentences):
    """
    Remove duplicate or near-duplicate sentences.
    """
    seen = set()
    unique = []

    for s in sentences:
        # Normalize for comparison
        normalized = re.sub(r"\s+", " ", s.lower().strip())
        if normalized not in seen:
            seen.add(normalized)
            unique.append(s)

    return unique


def extract_best_sentences(question, docs, embedding_model, top_n=5):
    """
    From retrieved chunks, find the most relevant sentences
    by comparing embeddings to the question.
    """

    # Step 1: Combine all chunk text
    full_text = " ".join([doc.page_content for doc in docs])

    # Step 2: Split into proper sentences (handles numbered lists)
    sentences = clean_and_split_sentences(full_text)

    # Step 3: Remove duplicates BEFORE scoring
    sentences = remove_duplicates(sentences)

    if not sentences:
        return "No relevant content found."

    # Step 4: Embed question and sentences
    q_embedding = embedding_model.embed_query(question)
    s_embeddings = embedding_model.embed_documents(sentences)

    # Step 5: Score each sentence
    scores = cosine_similarity([q_embedding], s_embeddings)[0]

    # Step 6: Pick top_n sentences by score
    top_n = min(top_n, len(sentences))
    top_indices = np.argsort(scores)[::-1][:top_n]

    # Step 7: Re-sort by original order for natural reading flow
    top_indices_sorted = sorted(top_indices)
    best_sentences = [sentences[i] for i in top_indices_sorted]

    # Step 8: Join into final answer
    answer = " ".join(best_sentences)

    return answer.strip()