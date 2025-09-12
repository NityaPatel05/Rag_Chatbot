# Advanced RAG System

An enterprise-grade document Q&A system built with **LangChain**, **Ollama**, and **Streamlit** featuring advanced retrieval techniques including merger retrieval, contextual compression, parent document retrieval, and HyDE.

## Key Features

### Advanced Retrieval Techniques

- **Merger Retriever**: Combines multiple retrieval methods for comprehensive results
- **Long Context Reorder**: Handles the "Lost in Middle Phenomenon" for better context utilization
- **Contextual Compression Retriever**: Compresses retrieved content while preserving relevance
- **Self Querying Retriever**: Interprets user queries to determine appropriate metadata filters
- **Parent Document Retriever**: Retrieves larger parent documents based on relevant chunks
- **HyDE (Hypothetical Document Embeddings)**: Uses generated hypothetical documents for better matching
- **Cross-Encoder Reranking**: Neural reranking for improved relevance scoring

### Smart Query Processing

- **Automatic Strategy Selection**: Chooses optimal retrieval strategy based on query type
- **Query Enhancement**: Expands queries with relevant context and synonyms
- **Multi-level Chunking**: Different chunk sizes for different retrieval strategies

### Smart Document Processing

- **Multi-format Support**: PDF, TXT, CSV files
- **Intelligent Chunking**: Recursive text splitting with overlap
- **Metadata Preservation**: Source tracking and page numbers

### Professional Interface

- **Streamlit Web UI**: Clean, enterprise-grade interface
- **Strategy Selection**: Manual or automatic retrieval strategy selection
- **Real-time Processing**: Live document processing and query handling
- **Source Attribution**: Proper citation tracking with expandable source views

## Quick Start

### 1. Prerequisites

Go to Ollama website and download the Windows installer:
https://ollama.ai/download

Open Command Prompt or PowerShell and run:
ollama serve

### 2. Setup Project

# Clone or download the project files

# Create virtual environment (recommended)

python -m venv venv
source venv\Scripts\activate

# Install dependencies

pip install -r requirements.txt

````

### 3. Configure Environment

Create a `.env` file in the project root:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large:latest

# Document Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Retrieval Configuration
RETRIEVER_K=8
COMPRESSION_RETRIEVER_K=10
HYDE_RETRIEVER_K=8
BM25_RETRIEVER_K=8
RERANK_TOP_K=5
FINAL_DOCS_COUNT=6

# Model Configuration
LLM_TEMPERATURE=0.1
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
````

### 4. Pull Required Models

# Essential models for the RAG system

ollama pull gemma3:1b # Main language model
ollama pull mxbai-embed-large:latest # Embedding model

```

### 5. Verify Setup


# Check Ollama is running
http://localhost:11434

# Verify models are installed
ollama list
```

### 5. Run the Application

# Activate virtual environment (if using)

source venv\Scripts\activate

# Start the application

streamlit run app.py

```

Open your browser and navigate to `http://localhost:8501`

## 📋 How to Use

### 1. Upload Documents

- Use the sidebar to upload PDF, TXT, or CSV files
- Click "Process Documents" to index them
- Wait for confirmation message

### 2. Ask Questions

- Type questions in the chat interface
- Get answers with inline citations [1], [2]
- View source documents at the bottom

### 3. Example Queries

- **Factual**: "What is the main topic of document 1?"
- **How-to**: "How to implement the solution mentioned?"
- **Comparison**: "Compare the approaches in different documents"
- **Analysis**: "Summarize the key findings"

## Architecture

### Advanced Retrieval Pipeline

```

Query Input
↓
Query Enhancement & Strategy Selection
↓
Multi-Strategy Retrieval:
├── Merger Retriever (Vector + BM25 + HyDE)
├── Contextual Compression
├── Parent Document Retrieval
└── Self-Query Filtering
↓
Cross-Encoder Reranking
↓
Long Context Reorder
↓
Context Formation & LLM Generation

````

### Retrieval Strategies

- **Factual Queries**: Contextual Compression for precise answers
- **Procedural Questions**: Parent Document Retrieval for comprehensive context
- **Comparison Tasks**: Merger Retriever for diverse perspectives
- **Analytical Queries**: HyDE for conceptual matching
- **General Questions**: Auto-selection of optimal strategy

### Components

- **LangChain**: Document processing and chain orchestration
- **Ollama**: Local LLM inference (gemma3:1b ) with connection validation
- **ChromaDB**: Dual vector databases for different chunk sizes
- **BM25**: Keyword-based retrieval
- **Cross-Encoder**: Neural reranking model
- **Streamlit**: Professional web interface
- **Modular Architecture**: Separated concerns for better maintainability

## ⚙ Configuration Options

### Model Settings

- `OLLAMA_MODEL`: Main language model (default: gemma3:1b )
- `OLLAMA_EMBEDDING_MODEL`: Embedding model (default: mxbai-embed-large:latest )
- `OLLAMA_BASE_URL`: Ollama server URL

### Configuration Settings

- `CHUNK_SIZE`: Document chunk size (default: 500)
- `CHUNK_OVERLAP`: Chunk overlap (default: 50)
- `RETRIEVER_K`: Number of documents to retrieve (default: 8)

    ## Troubleshooting

    **Missing Dependencies**

   # Activate virtual environment on Windows
venv\Scripts\activate

# Upgrade and install dependencies
pip install --upgrade -r requirements.txt

    **Import Errors**

   # Navigate to project folder
cd path\to\rag-chatbot

# Check Python paths
python -c "import sys; print(sys.path)"

    ```

    **Memory Issues**

    - Reduce `CHUNK_SIZE` in .env
    - Use smaller models: `ollama pull gemma3:1b `

    ### Performance Tips

    - Use SSD storage for ChromaDB
    - Increase `RETRIEVER_K` for better recall
    - Adjust chunk size based on document type

## Project Structure

````

rag-chatbot/
├── app.py # Main Streamlit application
├── chatbot.py # Core RAG chatbot class
├── config.py # Configuration and environment variables
├── models.py # LLM and embedding model management
├── document_processor.py # Document loading and chunking
├── retrieval_engine.py # Advanced retrieval strategies
├── retrievers.py # Custom retrievers (HyDE)
├── utils.py # Utility functions
├── requirements.txt # Python dependencies
├── .env # Environment variables
├── .gitignore # Git ignore patterns
├── README.md # This file
├── **pycache**/ # Python cache files (auto-generated)
├── chroma_db_small/ # Small chunks vector database
├── chroma_db_parent/ # Parent documents vector database
└── venv/ # Virtual environment (if using venv)

```

### Module Description

- **`app.py`**: Streamlit web interface and user interaction
- **`chatbot.py`**: Main orchestrator that coordinates all components
- **`config.py`**: Centralized configuration management with environment variables
- **`models.py`**: Manages LLM, embeddings, and cross-encoder models with connection validation
- **`document_processor.py`**: Handles document loading, text splitting, and metadata management
- **`retrieval_engine.py`**: Implements advanced retrieval strategies and document ranking
- **`retrievers.py`**: Custom retriever implementations (HyDE, etc.)
- **`utils.py`**: Helper functions for query processing and document formatting

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review Ollama documentation
3. Check LangChain documentation
4. Create an issue in the repository

---
```
