import asyncio
import os
# import ssl
# import certifi

from typing import Any, Dict, List
from dotenv import load_dotenv

# Import Langchain and other necessary modules here
# from langchain_core.agents import AgentExecutor
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from openai import OpenAI
from IPython.display import display, Markdown

load_dotenv()

def main():
    result = "# Heading\n\n**Bold text** and *italic*"
    print(result)