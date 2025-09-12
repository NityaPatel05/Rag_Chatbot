"""
Model initialization module for Advanced RAG System.
Contains LLM and embedding model setup.
"""

import streamlit as st
import requests
import time
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from sentence_transformers import CrossEncoder
from config import Config

class ModelManager:
    """Manages LLM, embeddings, and cross-encoder models."""
    
    def __init__(self):
        """Initialize all models."""
        self.llm = None
        self.embeddings = None
        self.cross_encoder = None
        self.setup_models()
    
    def setup_models(self):
        """Setup LLM, embeddings, and cross-encoder."""
        try:
            if not self._check_ollama_connection():
                st.error(f"Cannot connect to Ollama at {Config.OLLAMA_BASE_URL}. Please ensure Ollama is running.")
                return
            
            if not self._check_ollama_models():
                return
            
            self.llm = OllamaLLM(
                model=Config.OLLAMA_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=Config.LLM_TEMPERATURE
            )
            
            self.embeddings = OllamaEmbeddings(
                model=Config.OLLAMA_EMBEDDING_MODEL,
                base_url=Config.OLLAMA_BASE_URL
            )
            
            self.cross_encoder = CrossEncoder(Config.CROSS_ENCODER_MODEL)
            
            st.success("Components initialized successfully")
            
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            st.error("Please check if Ollama is running and the required models are installed.")
    
    def get_llm(self):
        """Get the LLM instance."""
        return self.llm
    
    def get_embeddings(self):
        """Get the embeddings instance."""
        return self.embeddings
    
    def get_cross_encoder(self):
        """Get the cross-encoder instance."""
        return self.cross_encoder
    
    def _check_ollama_connection(self):
        """Check if Ollama service is running."""
        try:
            response = requests.get(f"{Config.OLLAMA_BASE_URL}", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_ollama_models(self):
        """Check if required Ollama models are available."""
        try:
            response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=10)
            if response.status_code != 200:
                st.error("Failed to get available models from Ollama")
                return False
            
            available_models = [model['name'] for model in response.json().get('models', [])]
            
            if Config.OLLAMA_MODEL not in available_models:
                st.error(f"Model '{Config.OLLAMA_MODEL}' not found. Available models: {available_models}")
                st.info(f"Please run: ollama pull {Config.OLLAMA_MODEL}")
                return False
            
            if Config.OLLAMA_EMBEDDING_MODEL not in available_models:
                st.error(f"Embedding model '{Config.OLLAMA_EMBEDDING_MODEL}' not found. Available models: {available_models}")
                st.info(f"Please run: ollama pull {Config.OLLAMA_EMBEDDING_MODEL}")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Error checking Ollama models: {str(e)}")
            return False
    
    def is_ready(self):
        """Check if all models are ready."""
        return (self.llm is not None and 
                self.embeddings is not None and 
                self.cross_encoder is not None)
