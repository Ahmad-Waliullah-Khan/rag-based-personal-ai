import re
from langchain.llms import Ollama

def split_chapters(text: str):
    return re.split(r"\nChapter\s+\d+.*\n", text, flags=re.IGNORECASE)

def summarize_chunk(chunk, model="phi3:mini"):
    llm = Ollama(model=model)
    prompt = f"Summarize this chapter in bullet points:\n\n{chunk[:3000]}"
    return llm(prompt)
