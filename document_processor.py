"""
Document processing module for the Advanced RAG System.
Handles document loading, splitting, and metadata management.
"""

import tempfile
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import Config

class DocumentProcessor:
    """Handles document processing and chunking."""
    
    def __init__(self):
        """Initialize document processor with splitters."""
        self.setup_splitters()
    
    def setup_splitters(self):
        """Setup different text splitters for various chunk sizes."""
        self.small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        self.medium_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.MEDIUM_CHUNK_SIZE,
            chunk_overlap=Config.MEDIUM_CHUNK_OVERLAP
        )
        
        self.large_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.LARGE_CHUNK_SIZE,
            chunk_overlap=Config.LARGE_CHUNK_OVERLAP
        )
    
    def load_document(self, uploaded_file) -> List[Document]:
        """Load a single document from uploaded file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_file_path, encoding='utf-8')
            elif uploaded_file.name.endswith('.csv'):
                loader = CSVLoader(tmp_file_path)
            else:
                return []
            
            docs = loader.load()
            
            for doc in docs:
                doc.metadata.update({
                    'source_file': uploaded_file.name,
                    'file_type': uploaded_file.name.split('.')[-1],
                    'content_length': len(doc.page_content)
                })
            
            return docs
            
        except Exception:
            return []
        finally:
            os.unlink(tmp_file_path)
    
    def process_multiple_documents(self, uploaded_files) -> List[Document]:
        """Process multiple uploaded documents."""
        all_documents = []
        
        for uploaded_file in uploaded_files:
            docs = self.load_document(uploaded_file)
            all_documents.extend(docs)
        
        return all_documents
    
    def create_chunks(self, documents: List[Document]) -> tuple:
        """Create different sized chunks from documents."""
        small_chunks = self.small_splitter.split_documents(documents)
        medium_chunks = self.medium_splitter.split_documents(documents)
        large_chunks = self.large_splitter.split_documents(documents)
        
        return small_chunks, medium_chunks, large_chunks
    
    def get_small_splitter(self):
        """Get the small splitter instance."""
        return self.small_splitter
    
    def get_medium_splitter(self):
        """Get the medium splitter instance."""
        return self.medium_splitter
    
    def get_large_splitter(self):
        """Get the large splitter instance."""
        return self.large_splitter
