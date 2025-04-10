import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
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
import re
import time
from main import scrape_with_requests,handle_query,setup_rag_chain
import os



load_dotenv()


st.set_page_config(layout= "centered")
st.write("DeepMentor")
url = st.text_input("paste a url or link below:", placeholder = "https://example.com")

if st.button("Process URL"):
    if url:
        try:
            scraped = scrape_with_requests(url)
            st.success("Scraping successful.")
            st.session_state["scraped_content"] = scraped.text
            #st.markdown(scraped.markdown)
            st.session_state["rag_chain"] = setup_rag_chain(scraped.text)
        except Exception as e:
            st.error(f"Error processing the URL: {e}")
    else:
        st.error("Please enter a valid URL.")
        
st.subheader("💬 Chat with the Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ask anything about the link...")

if user_prompt:
    st.session_state.messages.append({"role":"user", "content":user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    if "scraped_content" in st.session_state and "rag_chain" in st.session_state:
        session_id = "web_rag_session"
        response = handle_query(
            st.session_state["rag_chain"],
            session_id= session_id
        )
        response = response.answer
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        st.error("Please process a URL before asking questions.")

