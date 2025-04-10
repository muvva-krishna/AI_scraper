import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import html2text
import json
import streamlit as st
import re
import time
from datetime import datetime
from pydantic import BaseModel, HttpUrl
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import retrieval_qa
from langchain.schema import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

class Scrapedcontent(BaseModel):
    url : HttpUrl
    html: str = None
    markdown: str = None
    text : str = None

class ChunkedDocument(BaseModel):
    chunk_id :int
    content: str
    
class RAGInput(BaseModel):
    answer: str

class RAGOutput(BaseModel):
    answer: str


def selenium_setup():
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service("C:\\Users\\krish\\Desktop\\AI_scraper\\chromedriver-win64\\chromedriver.exe"), options=options)
    return driver


def scrape_with_requests(url: str) -> Scrapedcontent:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises HTTPError if bad response

        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        converter = html2text.HTML2Text()
        markdown = converter.handle(str(soup))

        return Scrapedcontent(url=url, html=html, markdown=markdown, text=text)

    except Exception as e:
        print(f"Error occurred: {e}")
        raise e


def create_vectorstore(text :str):
    splitter = RecursiveCharacterTextSplitter(chunk_size =400, chunk_overlap = 100)
    chunks = splitter.split_text(text)
    documents = [ChunkedDocument(chunk_id=i, content=c) for i, c in enumerate(chunks)]
    raw_chunks = [doc.content for doc in documents]
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(raw_chunks, embedding=embeddings)
    return vectorstore

llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.2)
output_parser = StrOutputParser()

system_prompt = (
    "You are an assistant designed to answer questions based strictly on the provided context extracted from a website.\n\n"
    "You are NOT allowed to use any outside knowledge, prior training, or assumptions.\n\n"
    "If the answer is not clearly present in the given context, reply with:\n"
    "\"I'm sorry, the answer is not available in the provided content.\"\n\n"
    "Always be accurate, concise, and directly reference the relevant parts of the context to support your answers.\n\n"
    "Context:\n"
    "{context}"
)


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

main_prompt  = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])

def setup_rag_chain(text: str):
    vectorstore = create_vectorstore(text)
    retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs ={"k" :4})
    history_aware_retriever = create_history_aware_retriever(llm,retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm,main_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    memory_store ={}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in memory_store:
            memory_store[session_id] = ChatMessageHistory()
        return memory_store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )
    return conversational_rag_chain


def handle_query(rag_chain, input_prompt: str, session_id: str) -> RAGOutput:
    response = rag_chain.invoke(
        {"input": input_prompt},
        {"configurable": {"session_id": session_id}}
    )
    return RAGOutput(answer=response["answer"])
    
# Continuous interaction loop
if __name__ == "__main__":
    session_id = "unique_session_identifier"
    url = input("Paste the url tp scrape: ")
    scraped = scrape_with_requests(url)
    rag_chain = setup_rag_chain(scraped.text)
    while True:
        user_input = input("You| ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Ending the conversation. Goodbye!")
            break
        result = handle_query(rag_chain,user_input, session_id)
        print("Assistant |", result.answer)
        
        
