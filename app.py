import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
import streamlit as st
from langchain.schema import SystemMessage

# Load environment variables
load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool to search
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
    })

    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

search("what is meta's threads product about?")

# 2. Tool for scraping
def scrape_website(object: str, url: str):
    # This function will scrape website, and also will summarize the content based on the objective of the content
    # The objective is the original objective & task that the user gives to the agent, url is the url of the website to scrape

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }

    # Define the data to be sent in the request
    data = {
        "url": url,
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Sent the Post request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT: ", text)

        return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

scrape_website("what is langchain?", "https://python.langchain.com/en/latest/index.html")


# 3. Create a langchain agent with the tools above