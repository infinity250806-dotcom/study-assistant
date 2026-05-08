import re
from pypdf import PdfReader

def clean_text(text):

    # Remove excessive newlines
    text = text.replace("\n", " ")

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove repeated words
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    return text.strip()

def load_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted + " "

    return clean_text(text)