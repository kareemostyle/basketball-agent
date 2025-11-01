import asyncio
import os
import ssl
import certifi

from typing import Any, Dict, List
from dotenv import load_dotenv

# Import Langchain and other necessary modules here
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

load_dotenv()

async def main():
    """Main asynchronous function to run the ingestion process."""



if __name__ == "__main__":
    asyncio.run(main())