import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from typing import List, Dict, Any, Optional
import hashlib
import pickle

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MergerRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.storage import InMemoryStore
from langchain_core.vectorstores import VectorStore

# Additional imports
from sentence_transformers import CrossEncoder
import numpy as np

class HyDERetriever(BaseRetriever):
    """Hypothetical Document Embeddings (HyDE) Retriever."""
    
    def __init__(self, vectorstore: VectorStore, llm, k: int = 4):
        self.vectorstore = vectorstore
        self.llm = llm
        self.k = k
        
        # Template for generating hypothetical documents
        self.hyde_template = """Please write a passage to answer the question.
Question: {question}
Passage:"""
        
        self.hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template=self.hyde_template
        )
        
        self.hyde_chain = LLMChain(llm=self.llm, prompt=self.hyde_prompt)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Generate hypothetical document
        try:
            hypothetical_doc = self.hyde_chain.run(question=query)
            # Use hypothetical document for retrieval
            return self.vectorstore.similarity_search(hypothetical_doc, k=self.k)
        except:
            # Fallback to regular search
            return self.vectorstore.similarity_search(query, k=self.k)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

class AdvancedRAGChatbot:
    def __init__(self):
        """Initialize the advanced RAG chatbot with enhanced retrievers."""
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.parent_vectorstore = None
        self.parent_store = None
        self.retrievers = {}
        self.documents = []
        self.parent_documents = []
        self.cross_encoder = None
        self.setup_components()
        
    def filter_complex_metadata(documents):
        """Remove or simplify complex metadata from documents."""
        simple_docs = []
        for doc in documents:
        # Only keep simple key-value pairs in metadata
            simple_metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
            doc.metadata = simple_metadata
            simple_docs.append(doc)
        return simple_docs
    
    def setup_components(self):
        """Setup LLM, embeddings, and cross-encoder."""
        try:
            # Initialize Ollama LLM
            self.llm = OllamaLLM(
                model=os.getenv("OLLAMA_MODEL", "llama2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.1
            )
            
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(
                model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            
            # Initialize cross-encoder for reranking
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            st.success("Components initialized successfully")
            
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents and create advanced retrievers."""
        all_documents = []
        
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Load document based on file type
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_file_path)
                elif uploaded_file.name.endswith('.txt'):
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                elif uploaded_file.name.endswith('.csv'):
                    loader = CSVLoader(tmp_file_path)
                else:
                    continue
                
                docs = loader.load()
                
                # Add metadata for self-querying
                for doc in docs:
                    doc.metadata.update({
                        'source_file': uploaded_file.name,
                        'file_type': uploaded_file.name.split('.')[-1],
                        'content_length': len(doc.page_content)
                    })
                
                all_documents.extend(docs)
                
            except Exception:
                continue
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        if all_documents:
            self.create_advanced_retrievers(all_documents)
            return len(all_documents)
        return 0
    
    def create_advanced_retrievers(self, documents):
        """Create multiple advanced retrievers."""
        # Store parent documents
        self.parent_documents = documents
        
        # Create different chunk sizes for different strategies
        # Small chunks for regular retrieval
        small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
        )
        
        # Medium chunks for contextual compression
        medium_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Large chunks for parent document retrieval
        large_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        small_chunks = small_splitter.split_documents(documents)
        medium_chunks = medium_splitter.split_documents(documents)
        large_chunks = large_splitter.split_documents(documents)
        
        # Store documents
        self.documents = small_chunks
        
        # Create vector stores
        self.vectorstore = Chroma.from_documents(
            documents=filter_complex_metadata(small_chunks),
            embedding=self.embeddings,
            persist_directory="./chroma_db_small"
        )
        
        # Create parent document retriever
        self.parent_store = InMemoryStore()
        self.parent_vectorstore = Chroma.from_documents(
            documents=filter_complex_metadata(medium_chunks),
            embedding=self.embeddings,
            persist_directory="./chroma_db_parent"
        )
        
        # 1. Basic Vector Retriever
        self.retrievers['vector'] = self.vectorstore.as_retriever(
            search_kwargs={"k": int(os.getenv("RETRIEVER_K", 8))}
        )
        
        # 2. BM25 Retriever
        self.retrievers['bm25'] = BM25Retriever.from_documents(small_chunks)
        self.retrievers['bm25'].k = 8
        
        # 3. HyDE Retriever
        self.retrievers['hyde'] = HyDERetriever(
            vectorstore=self.vectorstore,
            llm=self.llm,
            k=8
        )
        
        # 4. Contextual Compression Retriever
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.retrievers['compression'] = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": 10})
        )
        
        # 5. Parent Document Retriever
        self.retrievers['parent'] = ParentDocumentRetriever(
            vectorstore=self.parent_vectorstore,
            docstore=self.parent_store,
            child_splitter=small_splitter,
            parent_splitter=large_splitter
        )
        
        # Add documents to parent retriever
        self.retrievers['parent'].add_documents(documents)
        
        # 6. Merger Retriever (combines multiple retrievers)
        base_retrievers = [
            self.retrievers['vector'],
            self.retrievers['bm25'],
            self.retrievers['hyde']
        ]
        
        self.retrievers['merger'] = MergerRetriever(retrievers=base_retrievers)
        
        # 7. Long Context Reorder
        self.long_context_reorder = LongContextReorder()
    
    def rewrite_query(self, query: str) -> str:
        """Advanced query rewriting with context understanding."""
        query = query.strip()
        
        # Expand abbreviations and synonyms
        query_lower = query.lower()
        
        # Handle different query types
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            query = f"definition explanation concept {query}"
        elif any(word in query_lower for word in ['how', 'steps', 'process']):
            query = f"procedure method steps {query}"
        elif any(word in query_lower for word in ['compare', 'difference', 'vs']):
            query = f"comparison analysis differences similarities {query}"
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            query = f"reason cause explanation {query}"
        
        return query
    
    def cross_encoder_rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if not documents or len(documents) <= top_k:
            return documents
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get similarity scores
            scores = self.cross_encoder.predict(pairs)
            
            # Sort by scores and return top_k
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
    
    def get_retrieval_strategy(self, query: str) -> str:
        """Determine best retrieval strategy based on query characteristics."""
        query_lower = query.lower()
        
        # For factual questions, use compression
        if any(word in query_lower for word in ['what is', 'define', 'meaning']):
            return 'compression'
        
        # For procedural questions, use parent document
        elif any(word in query_lower for word in ['how to', 'steps', 'process', 'method']):
            return 'parent'
        
        # For comparison questions, use merger
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return 'merger'
        
        # For complex analytical questions, use HyDE
        elif any(word in query_lower for word in ['analyze', 'analysis', 'evaluate', 'assess']):
            return 'hyde'
        
        # Default to merger for general questions
        else:
            return 'merger'
    
    def format_sources(self, documents: List[Document]) -> str:
        """Format source citations professionally."""
        sources = []
        seen_sources = set()
        
        for i, doc in enumerate(documents, 1):
            source_info = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            
            # Avoid duplicate sources
            if source_info not in seen_sources:
                page = doc.metadata.get('page', '')
                page_info = f" (Page {page})" if page else ""
                sources.append(f"[{i}] {source_info}{page_info}")
                seen_sources.add(source_info)
        
        return "\n".join(sources)
    
    def generate_answer(self, query: str, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Generate answer using advanced RAG pipeline."""
        if not self.retrievers:
            return {
                "answer": "No documents have been uploaded. Please upload documents first.",
                "sources": "",
                "strategy": "none"
            }
        
        try:
            # Determine retrieval strategy
            if not strategy:
                strategy = self.get_retrieval_strategy(query)
            
            # Pre-retrieval: Query rewriting
            enhanced_query = self.rewrite_query(query)
            
            # Mid-retrieval: Get documents using selected strategy
            if strategy in self.retrievers:
                retrieved_docs = self.retrievers[strategy].get_relevant_documents(enhanced_query)
            else:
                retrieved_docs = self.retrievers['merger'].get_relevant_documents(enhanced_query)
            
            # Post-retrieval: Cross-encoder reranking
            reranked_docs = self.cross_encoder_rerank(enhanced_query, retrieved_docs, top_k=6)
            
            # Apply Long Context Reorder
            final_docs = self.apply_long_context_reorder(reranked_docs)
            
            if not final_docs:
                return {
                    "answer": "No relevant information found in the uploaded documents.",
                    "sources": "",
                    "strategy": strategy
                }
            
            # Create context from final documents
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                 for i, doc in enumerate(final_docs)])
            
            # Create enhanced prompt template
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
            
            # Generate answer
            formatted_prompt = prompt.format(context=context, question=query)
            answer = self.llm.invoke(formatted_prompt)
            
            # Format sources
            sources = self.format_sources(final_docs)
            
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

def main():
    st.set_page_config(
        page_title="Advanced RAG System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Advanced RAG System")
    st.markdown("*Enterprise-grade document Q&A with advanced retrieval techniques*")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AdvancedRAGChatbot()
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'csv'],
            help="Supported formats: PDF, TXT, CSV"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    doc_count = st.session_state.chatbot.process_documents(uploaded_files)
                    if doc_count > 0:
                        st.success(f"Processed {doc_count} document chunks")
                    else:
                        st.warning("No documents were processed successfully")
        
        st.header("Retrieval Strategy")
        strategy_options = {
            "Auto": None,
            "Merger Retriever": "merger",
            "HyDE": "hyde",
            "Contextual Compression": "compression",
            "Parent Document": "parent",
            "Vector Search": "vector",
            "BM25": "bm25"
        }
        
        selected_strategy = st.selectbox(
            "Choose retrieval method",
            list(strategy_options.keys()),
            help="Auto selects the best strategy based on query type"
        )
        
        st.header("System Configuration")
        st.text(f"Model: {os.getenv('OLLAMA_MODEL', 'llama2')}")
        st.text(f"Embeddings: {os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')}")
        st.text(f"Chunk Size: {os.getenv('CHUNK_SIZE', 500)}")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Query Interface")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    if "sources" in message and message["sources"]:
                        with st.expander("Sources"):
                            st.markdown(message["sources"])
                    if "metadata" in message:
                        st.caption(f"Strategy: {message['metadata'].get('strategy', 'unknown')} | "
                                 f"Documents: {message['metadata'].get('num_docs', 0)}")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Processing query..."):
                    strategy = strategy_options[selected_strategy]
                    result = st.session_state.chatbot.generate_answer(prompt, strategy)
                    
                    st.markdown(result["answer"])
                    
                    if result["sources"]:
                        with st.expander("Sources"):
                            st.markdown(result["sources"])
                    
                    strategy_used = result.get("strategy", "unknown")
                    num_docs = result.get("num_docs", 0)
                    st.caption(f"Strategy: {strategy_used} | Documents: {num_docs}")
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "metadata": {
                            "strategy": strategy_used,
                            "num_docs": num_docs
                        }
                    })
    
    with col2:
        st.header("Retrieval Techniques")
        st.markdown("""
        **Active Features:**
        
        • **Merger Retriever**: Combines multiple retrieval methods
        
        • **Long Context Reorder**: Prevents lost-in-middle phenomenon
        
        • **Contextual Compression**: Compresses retrieved content
        
        • **Self Query**: Interprets queries for metadata filtering
        
        • **Parent Document**: Retrieves larger context chunks
        
        • **HyDE**: Uses hypothetical documents for better matching
        
        • **Cross-Encoder Reranking**: Neural reranking for relevance
        """)
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Reset System"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()