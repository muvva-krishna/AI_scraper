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

load_dotenv()

class Scrapedcontent(BaseModel):
    url : HttpUrl
    html: str = None
    markdown: str = None
    text : str = None




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





