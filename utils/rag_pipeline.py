from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_text(text)

    # Remove empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Remove duplicate chunks
    unique_chunks = []

    for chunk in chunks:
        if chunk not in unique_chunks:
            unique_chunks.append(chunk)

    return unique_chunks