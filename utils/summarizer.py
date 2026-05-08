from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

def generate_summary(text):

    prompt = f"""
    Summarize the following study material clearly:

    {text}

    Summary:
    """

    result = generator(
        prompt,
        max_new_tokens=120,
        truncation=True
    )

    generated_text = result[0]["generated_text"]

    summary = generated_text.split("Summary:")[-1]

    return summary.strip()