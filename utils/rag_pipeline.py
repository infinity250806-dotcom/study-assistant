from langchain_text_splitters import RecursiveCharacterTextSplitter
import anthropic


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


def generate_answer(question: str, docs: list) -> str:
    """Pass retrieved chunks + question to Claude to generate a proper answer."""

    # Build context from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in docs])

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a helpful study assistant. "
                    "Use ONLY the context below to answer the question. "
                    "If the answer is not in the context, say 'I could not find this in the uploaded document.'\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )

    return message.content[0].text