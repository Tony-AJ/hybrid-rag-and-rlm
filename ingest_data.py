"""
Unified Data Ingestion Script

Populates both ChromaDB and GraphDB with consistent data to ensure
hybrid retrieval works correctly (matching IDs across both systems).
"""

import os
import uuid
from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from graph_db.graph_db import GraphDatabase

# Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(CURRENT_DIR, "vector_db", "db", "chroma_db_with_metadata")
GRAPH_DB_PATH = os.path.join(CURRENT_DIR, "db", "graph_data.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ingest_data():
    """
    Ingest sample data into both Vector DB and Graph DB.
    
    Process:
    1. Define sample chunks with IDs
    2. Add to ChromaDB
    3. Add to GraphDB
    4. Create relationships in GraphDB
    """
    print("\n" + "="*60)
    print("UNIFIED DATA INGESTION")
    print("="*60 + "\n")
    
    # 1. Define Sample Data (Romeo & Juliet Summary Chunks)
    # We explicitly define IDs to ensure they match in both DBs
    data = [
        {
            "id": "chunk_1",
            "text": "The play is set in Verona, Italy, where two noble families, the Montagues and the Capulets, are locked in a violent feud.",
            "metadata": {"topic": "setting", "source": "summary"}
        },
        {
            "id": "chunk_2",
            "text": "Romeo Montague attends a Capulet ball in disguise, where he meets and falls in love with Juliet Capulet.",
            "metadata": {"topic": "meeting", "source": "summary"}
        },
        {
            "id": "chunk_3",
            "text": "Romeo and Juliet are secretly married by Friar Lawrence, who hopes their union will reconcile their warring families.",
            "metadata": {"topic": "marriage", "source": "summary"}
        },
        {
            "id": "chunk_4",
            "text": "Tybalt Capulet challenges Romeo to a duel. Romeo refuses, but his friend Mercutio fights instead and is killed.",
            "metadata": {"topic": "conflict", "source": "summary"}
        },
        {
            "id": "chunk_5",
            "text": "Enraged by Mercutio's death, Romeo kills Tybalt and is subsequently banished from Verona by the Prince.",
            "metadata": {"topic": "banishment", "source": "summary"}
        },
        {
            "id": "chunk_6",
            "text": "Juliet's father arranges for her to marry Count Paris. To avoid this, she takes a potion that makes her appear dead.",
            "metadata": {"topic": "plot", "source": "summary"}
        },
        {
            "id": "chunk_7",
            "text": "Romeo hears of Juliet's death but not the plan. He goes to her tomb, drinks poison, and dies by her side.",
            "metadata": {"topic": "tragedy", "source": "summary"}
        },
        {
            "id": "chunk_8",
            "text": "Juliet wakes up, finds Romeo dead, and stabs herself with his dagger. The families end their feud in shared grief.",
            "metadata": {"topic": "resolution", "source": "summary"}
        }
    ]
    
    # 2. Populate ChromaDB (Vector Store)
    print("Initializing ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
    
    # Create Document objects
    documents = []
    for item in data:
        doc = Document(
            page_content=item["text"],
            metadata={
                "id": item["id"],
                **item["metadata"]
            }
        )
        documents.append(doc)
    
    print(f"Adding {len(documents)} documents to Vector DB...")
    # We use the IDs argument to ensure Chroma uses our custom IDs
    ids = [item["id"] for item in data]
    vector_db.add_documents(documents, ids=ids)
    print(" Vector DB populated")
    
    # 3. Populate GraphDB (Graph Store)
    print("\nInitializing GraphDB...")
    graph_db = GraphDatabase(db_path=GRAPH_DB_PATH, auto_persist=True)
    
    # Clear existing graph for a clean state
    graph_db.graph.clear()
    graph_db._edge_id_map.clear()
    
    print("Adding nodes to Graph DB...")
    for item in data:
        graph_db.create_node(
            text=item["text"],
            metadata=item["metadata"],
            embedding=None  # We could add embeddings here too if needed
        )
        # Force the ID to match our schema (create_node generates UUID by default if not handled, 
        # but our GraphDB wrapper might not expose ID setting easily in create_node.
        # Let's check GraphDB implementation... 
        # It returns a node with generated ID. We need to swap it or use a method that allows setting ID.
        # Looking at graph_db.py, create_node doesn't take ID. 
        # But we can manually add/rename.
        
        # Actually, let's look at GraphNode model. It takes node_id.
        # But GraphDatabase.create_node instantiates GraphNode without passing ID.
        # So we have to manually fix it in the graph object as we did in setup_graph.py
        
        # Remove the auto-generated node (last added)
        # Actually, let's just use the graph object directly for precision
        
    # Re-implementing node addition to ensure IDs match
    for item in data:
        # We can use the method we used in setup_graph.py
        # Create a temporary node to get the object structure if needed, or just add directly
        
        # Direct NetworkX manipulation to ensure ID control
        graph_db.graph.add_node(
            item["id"],
            text=item["text"],
            metadata=item["metadata"],
            embedding=None
        )
        print(f"  Node: {item['id']}")

    # 4. Create Relationships
    print("\nCreating relationships...")
    relationships = [
        ("chunk_1", "chunk_2", "next", 1.0),
        ("chunk_2", "chunk_3", "next", 1.0),
        ("chunk_3", "chunk_4", "next", 1.0),
        ("chunk_4", "chunk_5", "next", 1.0),
        ("chunk_5", "chunk_6", "next", 1.0),
        ("chunk_6", "chunk_7", "next", 1.0),
        ("chunk_7", "chunk_8", "next", 1.0),
        
        # Semantic connections (non-linear)
        ("chunk_1", "chunk_4", "foreshadows", 0.5), # Feud -> Conflict
        ("chunk_3", "chunk_8", "goal_achieved", 0.8), # Marriage -> Resolution (reconciliation)
        ("chunk_2", "chunk_6", "character_arc", 0.6), # Meeting -> Plot
    ]
    
    for source, target, rel, weight in relationships:
        graph_db.create_edge(source, target, rel, weight)
        print(f"  Edge: {source} -> {target} ({rel})")
        
    graph_db.persist()
    print("\n Graph DB populated")
    
    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("Both databases are now synchronized with matching IDs.")
    print("="*60 + "\n")


if __name__ == "__main__":
    ingest_data()
