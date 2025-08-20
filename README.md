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
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve
```

### 2. Setup Project
```bash
# Clone or download the project files
# Install dependencies
pip install -r requirements.txt


```

### 3. Configure Environment
Edit `.env` file:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_K=10
```

### 4. Pull Required Models
```bash
ollama pull llama2
ollama pull nomic-embed-text
```

### 5. Run the Application
```bash
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
```

### Retrieval Strategies
- **Factual Queries**: Contextual Compression for precise answers
- **Procedural Questions**: Parent Document Retrieval for comprehensive context
- **Comparison Tasks**: Merger Retriever for diverse perspectives
- **Analytical Queries**: HyDE for conceptual matching
- **General Questions**: Auto-selection of optimal strategy

### Components
- **LangChain**: Document processing and chain orchestration
- **Ollama**: Local LLM inference (llama2)
- **ChromaDB**: Vector database for semantic search
- **BM25**: Keyword-based retrieval
- **Cross-Encoder**: Neural reranking model
- **Streamlit**: Web interface

## ⚙ Configuration Options

### Model Settings
- `OLLAMA_MODEL`: Main language model (default: llama2)
- `OLLAMA_EMBEDDING_MODEL`: Embedding model (default: nomic-embed-text)
- `OLLAMA_BASE_URL`: Ollama server URL

### Configuration Settings
- `CHUNK_SIZE`: Document chunk size (default: 500)
- `CHUNK_OVERLAP`: Chunk overlap (default: 50)  
- `RETRIEVER_K`: Number of documents to retrieve (default: 8)

##  Troubleshooting

### Common Issues

**Ollama Connection Error**
```bash
# Make sure Ollama is running
ollama serve

# Check if models are available
ollama list
```

**Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

**Memory Issues**
- Reduce `CHUNK_SIZE` in .env
- Use smaller models: `ollama pull llama2:7b`

### Performance Tips
- Use SSD storage for ChromaDB
- Increase `RETRIEVER_K` for better recall
- Adjust chunk size based on document type

##  Project Structure

```
rag-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── README.md          # This file
└── chroma_db/         # Vector database (created automatically)
```

##  Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

##  License

MIT License - feel free to modify and distribute!

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Ollama documentation
3. Check LangChain documentation
4. Create an issue in the repository

---

