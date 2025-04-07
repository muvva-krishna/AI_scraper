import os
import pandas as pd
import streamlit as st
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
import re
import time



load_dotenv()


st.set_page_config(layout= "centered")
st.write("DeepMentor")
url = st.text_input("paste a url or link below:", placeholder = "https://example.com")

if st.button("Process URL"):
    if url :
        st.success(f"URL recieved : {url}")
    else:
        st.error("Please a valid URL")
        
st.subheader("💬 Chat with the Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
user_prompt = st.chat_input("Ask anything about the link....")
if user_prompt:
    st.write(".......")
    st.session_state.messages.append({"role":"user", "content":user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    response = f"processing your question , based on the the URL :{url}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
