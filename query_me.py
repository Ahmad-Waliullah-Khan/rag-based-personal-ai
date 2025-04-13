# query_me.py

from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

llm = Ollama(model="mistral")
embedding = OllamaEmbeddings(model="mistral")
vectordb = Chroma(persist_directory="db", embedding_function=embedding)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

print("🧠 Ask me anything about your life!")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = qa.run(query)
    print("AI:", answer)
