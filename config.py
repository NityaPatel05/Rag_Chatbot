"""
Configuration module for Advanced RAG System.
Contains all environment variables and configuration settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the RAG system."""
    
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large:latest")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", 8))
    
    CHROMA_DB_SMALL = "./chroma_db_small"
    CHROMA_DB_PARENT = "./chroma_db_parent"
    
    CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    
    SUPPORTED_FILE_TYPES = ['pdf', 'txt', 'csv']
    
    LLM_TEMPERATURE = 0.1
    
    COMPRESSION_RETRIEVER_K = 10
    HYDE_RETRIEVER_K = 8
    BM25_RETRIEVER_K = 8
    RERANK_TOP_K = 5
    FINAL_DOCS_COUNT = 6
    
    MEDIUM_CHUNK_SIZE = 1000
    MEDIUM_CHUNK_OVERLAP = 100
    LARGE_CHUNK_SIZE = 2000
    LARGE_CHUNK_OVERLAP = 200
