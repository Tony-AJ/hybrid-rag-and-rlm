import os
# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List, Dict, Any, Optional, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

class VectorDatabase:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Disable telemetry to reduce noise
        settings = chromadb.config.Settings(anonymized_telemetry=False)
        
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
            client_settings=settings
        )

    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        """Add or update a document in the vector store."""
        self.db.add_texts(
            texts=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def add_documents(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        """Add multiple documents in batch."""
        if not ids:
            return
        self.db.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )

    def delete_document(self, doc_id: str):
        """Delete a document from the vector store."""
        try:
            self.db.delete(ids=[doc_id])
        except ValueError:
            pass

    def search(self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """
        Search for documents similar to the query.
        Returns: List of (id, text, score, metadata)
        """
        results = self.db.similarity_search_with_score(query, k=top_k, filter=filter)
        formatted_results = []
        for doc, score in results:
            similarity = 1.0 / (1.0 + score)
            doc_id = doc.metadata.get('id')
            formatted_results.append((doc_id, doc.page_content, similarity, doc.metadata))
            
        return formatted_results

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        """Update a document. In Chroma, adding with same ID overwrites."""
        self.add_document(doc_id, text, metadata)

