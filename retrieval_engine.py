"""
Retrieval engine module for the Advanced RAG System.
Manages multiple retrieval strategies and document ranking.
"""

from typing import List, Dict, Any
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MergerRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain.storage import InMemoryStore

from retrievers import HyDERetriever
from utils import filter_complex_metadata
from config import Config

class RetrievalEngine:
    """Manages advanced retrieval strategies and document ranking."""
    
    def __init__(self, model_manager, document_processor):
        """Initialize retrieval engine."""
        self.model_manager = model_manager
        self.document_processor = document_processor
        self.retrievers = {}
        self.vectorstore = None
        self.parent_vectorstore = None
        self.parent_store = None
        self.long_context_reorder = LongContextReorder()
    
    def create_retrievers(self, documents: List[Document]):
        """Create multiple advanced retrievers."""
        small_chunks, medium_chunks, large_chunks = self.document_processor.create_chunks(documents)
        
        self._create_vector_stores(small_chunks, medium_chunks)
        
        self._create_individual_retrievers(small_chunks, medium_chunks, documents)
        
        self._create_ensemble_retrievers()
    
    def _create_vector_stores(self, small_chunks: List[Document], medium_chunks: List[Document]):
        """Create vector stores for different chunk sizes."""
        self.vectorstore = Chroma.from_documents(
            documents=filter_complex_metadata(small_chunks),
            embedding=self.model_manager.get_embeddings(),
            persist_directory=Config.CHROMA_DB_SMALL
        )
        
        self.parent_store = InMemoryStore()
        self.parent_vectorstore = Chroma.from_documents(
            documents=filter_complex_metadata(medium_chunks),
            embedding=self.model_manager.get_embeddings(),
            persist_directory=Config.CHROMA_DB_PARENT
        )
    
    def _create_individual_retrievers(self, small_chunks: List[Document], medium_chunks: List[Document], documents: List[Document]):
        """Create individual retriever instances."""
        self.retrievers['vector'] = self.vectorstore.as_retriever(
            search_kwargs={"k": Config.RETRIEVER_K}
        )
        
        self.retrievers['bm25'] = BM25Retriever.from_documents(small_chunks)
        self.retrievers['bm25'].k = Config.BM25_RETRIEVER_K
        
        self.retrievers['hyde'] = HyDERetriever(
            vectorstore=self.vectorstore,
            llm=self.model_manager.get_llm(),
            k=Config.HYDE_RETRIEVER_K
        )
        
        compressor = LLMChainExtractor.from_llm(self.model_manager.get_llm())
        self.retrievers['compression'] = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": Config.COMPRESSION_RETRIEVER_K})
        )
        
    
        self.retrievers['parent'] = ParentDocumentRetriever(
            vectorstore=self.parent_vectorstore,
            docstore=self.parent_store,
            child_splitter=self.document_processor.get_small_splitter(),
            parent_splitter=self.document_processor.get_large_splitter()
        )
        
        self.retrievers['parent'].add_documents(documents)
    
    def _create_ensemble_retrievers(self):
        """Create ensemble retrievers that combine multiple strategies."""
        base_retrievers = [
            self.retrievers['vector'],
            self.retrievers['bm25'],
            self.retrievers['hyde']
        ]
        
        self.retrievers['merger'] = MergerRetriever(retrievers=base_retrievers)
    
    def retrieve_documents(self, query: str, strategy: str) -> List[Document]:
        """Retrieve documents using specified strategy."""
        if strategy in self.retrievers:
            return self.retrievers[strategy].invoke(query)
        else:
            return self.retrievers['merger'].invoke(query)
    
    def cross_encoder_rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if top_k is None:
            top_k = Config.RERANK_TOP_K
            
        if not documents or len(documents) <= top_k:
            return documents
        
        try:
            pairs = [[query, doc.page_content] for doc in documents]
            
            scores = self.model_manager.get_cross_encoder().predict(pairs)
            
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_docs[:top_k]]
        except:
            return documents[:top_k]
    
    def apply_long_context_reorder(self, documents: List[Document]) -> List[Document]:
        """Apply Long Context Reorder to handle lost in middle phenomenon."""
        try:
            return self.long_context_reorder.transform_documents(documents)
        except:
            return documents
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available retrieval strategies."""
        return list(self.retrievers.keys())
    
    def is_ready(self) -> bool:
        """Check if retrieval engine is ready with retrievers."""
        return bool(self.retrievers)
