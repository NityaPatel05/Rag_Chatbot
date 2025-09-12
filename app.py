"""
Streamlit UI for Advanced RAG System.
Main application interface for the document Q&A system.
"""

import streamlit as st
from typing import Optional

from chatbot import AdvancedRAGChatbot
from config import Config

def get_strategy_options():
    """Get available strategy options for the UI."""
    return {
        "Auto": None,
        "Merger Retriever": "merger",
        "HyDE": "hyde",
        "Contextual Compression": "compression",
        "Parent Document": "parent",
        "Vector Search": "vector",
        "BM25": "bm25"
    }

def main():
    st.set_page_config(
        page_title="Advanced RAG System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Advanced RAG System")
    st.markdown("*Enterprise-grade document Q&A with advanced retrieval techniques*")

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AdvancedRAGChatbot()

    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=Config.SUPPORTED_FILE_TYPES,
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
        strategy_options = get_strategy_options()
        
        selected_strategy = st.selectbox(
            "Choose retrieval method",
            list(strategy_options.keys()),
            help="Auto selects the best strategy based on query type"
        )
        
        st.header("System Configuration")
        st.text(f"Model: {Config.OLLAMA_MODEL}")
        st.text(f"Embeddings: {Config.OLLAMA_EMBEDDING_MODEL}")
        st.text(f"Chunk Size: {Config.CHUNK_SIZE}")

def main():
    """Main Streamlit application function."""
    st.set_page_config(
        page_title="Advanced RAG System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Advanced RAG System")
    st.markdown("*Enterprise-grade document Q&A with advanced retrieval techniques*")

    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing Advanced RAG System..."):
            st.session_state.chatbot = AdvancedRAGChatbot()
    
    if not st.session_state.chatbot.model_manager.is_ready():
        st.error("⚠️ Ollama Connection Issue")
        st.markdown("""
        **Please ensure:**
        1. Ollama is installed and running
        2. Required models are downloaded
        
        **Quick Setup:**
        ```bash
        ollama pull llama2
        ollama pull nomic-embed-text
        ollama serve
        ```
        
        See `OLLAMA_SETUP.md` for detailed instructions.
        """)
        st.stop()

    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=Config.SUPPORTED_FILE_TYPES,
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
        strategy_options = get_strategy_options()
        
        selected_strategy = st.selectbox(
            "Choose retrieval method",
            list(strategy_options.keys()),
            help="Auto selects the best strategy based on query type"
        )
        
        st.header("System Configuration")
        st.text(f"Model: {Config.OLLAMA_MODEL}")
        st.text(f"Embeddings: {Config.OLLAMA_EMBEDDING_MODEL}")
        st.text(f"Chunk Size: {Config.CHUNK_SIZE}")

    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Query Interface")

        if "messages" not in st.session_state:
            st.session_state.messages = []

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

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

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