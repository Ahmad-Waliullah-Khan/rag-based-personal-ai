from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = Ollama(model="mistral")  # Or llama3/phi3

template = """
You are my helpful personal assistant named FRIDAY..
Answer the following question:

{question}
"""

prompt = PromptTemplate(input_variables=["question"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

while True:
    q = input("Ask me anything: ")
    print(chain.run(q))
