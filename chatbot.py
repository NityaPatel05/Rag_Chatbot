
import streamlit as st
from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from models import ModelManager
from document_processor import DocumentProcessor
from retrieval_engine import RetrievalEngine
from utils import rewrite_query, get_retrieval_strategy, format_sources
from config import Config

class AdvancedRAGChatbot:
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.document_processor = DocumentProcessor()
        self.retrieval_engine = RetrievalEngine(self.model_manager, self.document_processor)
        self.documents = []
        self.parent_documents = []
    
    def process_documents(self, uploaded_files) -> int:
        if not self.model_manager.is_ready():
            st.error("Models are not ready. Please check Ollama connection and try again.")
            return 0
        
        all_documents = self.document_processor.process_multiple_documents(uploaded_files)
        
        if all_documents:
            self.parent_documents = all_documents
            
            self.retrieval_engine.create_retrievers(all_documents)
            
            return len(all_documents)
        return 0
    
    def generate_answer(self, query: str, strategy: Optional[str] = None) -> Dict[str, Any]:
        if not self.model_manager.is_ready():
            return {
                "answer": "Models are not ready. Please check Ollama connection and required models.",
                "sources": "",
                "strategy": "none"
            }
        
        if not self.retrieval_engine.is_ready():
            return {
                "answer": "No documents have been uploaded. Please upload documents first.",
                "sources": "",
                "strategy": "none"
            }
        
        try:
            if not strategy:
                strategy = get_retrieval_strategy(query)
            
            enhanced_query = rewrite_query(query)
            
            retrieved_docs = self.retrieval_engine.retrieve_documents(enhanced_query, strategy)
            
            reranked_docs = self.retrieval_engine.cross_encoder_rerank(
                enhanced_query, 
                retrieved_docs, 
                top_k=Config.FINAL_DOCS_COUNT
            )
            
            final_docs = self.retrieval_engine.apply_long_context_reorder(reranked_docs)
            
            if not final_docs:
                return {
                    "answer": "No relevant information found in the uploaded documents.",
                    "sources": "",
                    "strategy": strategy
                }
            
            answer = self._generate_llm_response(query, final_docs)
            
            sources = format_sources(final_docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "strategy": strategy,
                "num_docs": len(final_docs)
            }
            
        except Exception as e:
            return {
                "answer": "An error occurred while processing your question. Please try again.",
                "sources": "",
                "strategy": "error"
            }
    
    def _generate_llm_response(self, query: str, documents: List[Document]) -> str:
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(documents)
        ])
        
        prompt_template = """You are an Advanced RAG Assistant. Answer the user's question based strictly on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Base your answer only on the provided documents
2. Use inline citations like [1], [2] for specific claims
3. Structure your answer with clear headings and bullet points where appropriate
4. If the context doesn't contain enough information, state this clearly
5. Provide a comprehensive but concise answer
6. Do not add information not present in the context

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        formatted_prompt = prompt.format(context=context, question=query)
        answer = self.model_manager.get_llm().invoke(formatted_prompt)
        
        return answer
    
    def get_available_strategies(self) -> List[str]:
        if self.retrieval_engine.is_ready():
            return self.retrieval_engine.get_available_strategies()
        return []
    
    def is_ready(self) -> bool:
        return self.retrieval_engine.is_ready()
