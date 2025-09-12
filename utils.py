"""
Utility functions for the Advanced RAG System.
"""

from typing import List
from langchain_core.documents import Document

def filter_complex_metadata(documents: List[Document]) -> List[Document]:
    """Remove or simplify complex metadata from documents."""
    simple_docs = []
    for doc in documents:
        simple_metadata = {
            k: v for k, v in doc.metadata.items() 
            if isinstance(v, (str, int, float, bool))
        }
        doc.metadata = simple_metadata
        simple_docs.append(doc)
    return simple_docs

def format_sources(documents: List[Document]) -> str:
    """Format source citations professionally."""
    sources = []
    seen_sources = set()
    
    for i, doc in enumerate(documents, 1):
        source_info = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
        
        if source_info not in seen_sources:
            page = doc.metadata.get('page', '')
            page_info = f" (Page {page})" if page else ""
            sources.append(f"[{i}] {source_info}{page_info}")
            seen_sources.add(source_info)
    
    return "\n".join(sources)

def rewrite_query(query: str) -> str:
    """Advanced query rewriting with context understanding."""
    query = query.strip()
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['what', 'define', 'explain']):
        query = f"definition explanation concept {query}"
    elif any(word in query_lower for word in ['how', 'steps', 'process']):
        query = f"procedure method steps {query}"
    elif any(word in query_lower for word in ['compare', 'difference', 'vs']):
        query = f"comparison analysis differences similarities {query}"
    elif any(word in query_lower for word in ['why', 'reason', 'cause']):
        query = f"reason cause explanation {query}"
    
    return query

def get_retrieval_strategy(query: str) -> str:
    """Determine best retrieval strategy based on query characteristics."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['what is', 'define', 'meaning']):
        return 'compression'
    elif any(word in query_lower for word in ['how to', 'steps', 'process', 'method']):
        return 'parent'
    elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
        return 'merger'
    elif any(word in query_lower for word in ['analyze', 'analysis', 'evaluate', 'assess']):
        return 'hyde'
    else:
        return 'merger'
