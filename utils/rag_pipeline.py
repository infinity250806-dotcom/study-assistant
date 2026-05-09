from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import groq

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return list(dict.fromkeys(chunks))


def generate_answer(question: str, docs: list) -> str:
    context = "\n\n".join([doc.page_content for doc in docs])

    client = Groq()  # reads GROQ_API_KEY from env

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # free & powerful
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

    return response.choices[0].message.content