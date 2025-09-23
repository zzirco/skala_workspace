from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import OpenAIEmbeddings
import os
from db import CONN_STR

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = PGVector(
    connection_string=CONN_STR,
    embedding_function=embeddings,
    collection_name="copyright_cases"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

def ask_question(query: str) -> str:
    return qa_chain.run(query)
