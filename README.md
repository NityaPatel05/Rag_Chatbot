Advanced RAG System
An enterprise-grade document Q&A system built with LangChain, Ollama, and Streamlit featuring advanced retrieval techniques including merger retrieval, contextual compression, parent document retrieval, and HyDE.
Key Features
Advanced Retrieval Techniques

Merger Retriever: Combines multiple retrieval methods for comprehensive results
Long Context Reorder: Handles the "Lost in Middle Phenomenon" for better context utilization
Contextual Compression Retriever: Compresses retrieved content while preserving relevance
Self Querying Retriever: Interprets user queries to determine appropriate metadata filters
Parent Document Retriever: Retrieves larger parent documents based on relevant chunks
HyDE (Hypothetical Document Embeddings): Uses generated hypothetical documents for better matching
Cross-Encoder Reranking: Neural reranking for improved relevance scoring

Smart Query Processing

Automatic Strategy Selection: Chooses optimal retrieval strategy based on query type
Query Enhancement: Expands queries with relevant context and synonyms
Multi-level Chunking: Different chunk sizes for different retrieval strategies

Smart Document Processing

Multi-format Support: PDF, TXT, CSV files
Intelligent Chunking: Recursive text splitting with overlap
Metadata Preservation: Source tracking and page numbers

Professional Interface

Streamlit Web UI: Clean, enterprise-grade interface
Strategy Selection: Manual or automatic retrieval strategy selection
Real-time Processing: Live document processing and query handling
Source Attribution: Proper citation tracking with expandable source views

