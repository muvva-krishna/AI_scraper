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



load_dotenv()
st.write("DeepMentor")

def selenium_setup():
    options = Options()
    service = Service("chromedriver.exe")
    driver = webdriver.Chrome(service = service, options=options)
    return driver

def webscrape(url):
    driver = selenium_setup()

    try:
        driver.get(url)
        time.sleep(1)
        html = driver.page_source
        return html
    finally:
        driver.quit()
        
def parse_html(html_content):
    soup = str(BeautifulSoup(html_content,'html_parser'))
    return soup

def markdown_converter(html_content):
    parsed_html = parse_html(html_content)
    markdown_converter = html2text.HTML2Text()
    markdown_content = markdown_converter.handle(parsed_html)
    return markdown_content

