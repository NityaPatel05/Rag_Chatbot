"""
Custom retrievers for the Advanced RAG System.
"""

from typing import List, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from config import Config

class HyDERetriever(BaseRetriever):
    """Hypothetical Document Embeddings (HyDE) Retriever."""
    
    vectorstore: VectorStore = Field(description="Vector store for similarity search")
    llm: Any = Field(description="Language model for generating hypothetical documents")
    k: int = Field(default=8, description="Number of documents to retrieve")
    hyde_chain: Any = Field(default=None, description="LLM chain for HyDE generation")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, vectorstore: VectorStore, llm: Any, k: int = None, **kwargs):
        """Initialize HyDE retriever."""
        k = k or Config.HYDE_RETRIEVER_K
        
        hyde_template = """Please write a passage to answer the question.
Question: {question}
Passage:"""
        
        hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template=hyde_template
        )
        
        hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)
        
        super().__init__(
            vectorstore=vectorstore,
            llm=llm,
            k=k,
            hyde_chain=hyde_chain,
            **kwargs
        )
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents using hypothetical document embeddings."""
        try:
            hypothetical_doc = self.hyde_chain.run(question=query)
            
            return self.vectorstore.similarity_search(hypothetical_doc, k=self.k)
        except:
            return self.vectorstore.similarity_search(query, k=self.k)
