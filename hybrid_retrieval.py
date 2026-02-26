"""
Hybrid Retrieval System: ChromaDB (Vector Search) + NetworkX (Graph Relations)

End-to-end implementation combining cosine similarity from vector search
and graph-based relevance scoring for enhanced retrieval.
"""

import os
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import networkx as nx
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from graph_db.graph_db import GraphDatabase


# ==================== Configuration ====================

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(CURRENT_DIR, "vector_db", "db", "chroma_db_with_metadata")
GRAPH_DB_PATH = os.path.join(CURRENT_DIR, "db", "graph_data.json")

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Hybrid scoring weights
COSINE_WEIGHT = 0.7
GRAPH_WEIGHT = 0.3

# Graph traversal depth
GRAPH_DEPTH = 2


# ==================== Initialization ====================

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize ChromaDB vector store
chroma_db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

# Initialize Graph Database
graph_db = GraphDatabase(db_path=GRAPH_DB_PATH, auto_persist=True)


# ==================== 1. Vector Search ====================

def vector_search(query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """
    Perform vector search using ChromaDB with cosine similarity.
    
    Args:
        query: User query string
        top_k: Number of top results to return
        
    Returns:
        List of tuples: (chunk_id, chunk_text, cosine_score)
        
    Assumptions:
        - ChromaDB collection uses chunk index as metadata 'id' or generates IDs
        - Cosine similarity scores are normalized [0, 1]
    """
    # Use ChromaDB's similarity search with scores
    results = chroma_db.similarity_search_with_score(query, k=top_k)
    
    vector_results = []
    for doc, score in results:
        # Extract chunk_id from metadata (fallback to generated ID)
        chunk_id = doc.metadata.get('id', doc.metadata.get('source', str(hash(doc.page_content))))
        chunk_text = doc.page_content
        
        # ChromaDB returns distance, convert to similarity (1 - normalized_distance)
        # Assuming L2 distance, normalize to [0, 1] similarity
        cosine_score = 1.0 / (1.0 + score)  # Convert distance to similarity
        
        vector_results.append((chunk_id, chunk_text, cosine_score))
    
    return vector_results


# ==================== 2. Graph Expansion & Scoring ====================

def graph_score(graph: nx.MultiDiGraph, chunk_id: str, depth: int = GRAPH_DEPTH) -> Dict[str, float]:
    """
    Compute graph-based scores for nodes related to chunk_id.
    
    Args:
        graph: NetworkX MultiDiGraph instance
        chunk_id: Starting chunk/node ID
        depth: Maximum traversal depth (default: 2)
        
    Returns:
        Dictionary mapping related_chunk_id -> relation_score
        
    Score assignment:
        - depth 1 (direct neighbors): score 1.0
        - depth 2 (2-hop neighbors): score 0.5
        - no connection: score 0.0
        
    Assumptions:
        - Graph nodes represent chunk IDs
        - Edges represent meaningful relationships (parent-child, topic-subtopic, etc.)
    """
    if chunk_id not in graph.nodes:
        return {}
    
    scores = {}
    visited = set()
    queue = [(chunk_id, 0)]  # (node_id, current_depth)
    
    while queue:
        node_id, current_depth = queue.pop(0)
        
        if node_id in visited:
            continue
        
        visited.add(node_id)
        
        # Assign score based on depth
        if current_depth == 0:
            scores[node_id] = 1.0  # Starting node
        elif current_depth == 1:
            scores[node_id] = 1.0  # Direct neighbors
        elif current_depth == 2:
            scores[node_id] = 0.5  # 2-hop neighbors
        else:
            scores[node_id] = 0.0  # Beyond depth
        
        # Explore neighbors if within depth
        if current_depth < depth:
            # Get both outgoing and incoming neighbors
            neighbors = set(graph.successors(node_id)) | set(graph.predecessors(node_id))
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, current_depth + 1))
    
    return scores


# ==================== 3. Hybrid Scoring ====================

def hybrid_rank(
    cosine_results: List[Tuple[str, str, float]],
    graph_scores: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Combine cosine similarity and graph scores using hybrid formula.
    
    Formula:
        final_score = cosine_similarity * 0.7 + graph_score * 0.3
    
    Args:
        cosine_results: List of (chunk_id, chunk_text, cosine_score)
        graph_scores: Dictionary {chunk_id: graph_score}
        
    Returns:
        Sorted list of result dictionaries with final scores
    """
    # Collect all unique chunks from both sources
    chunk_data = {}  # chunk_id -> {text, cosine_score, graph_score}
    
    # Add vector search results
    for chunk_id, chunk_text, cos_score in cosine_results:
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'cosine_similarity': cos_score,
                'graph_score': 0.0
            }
        else:
            chunk_data[chunk_id]['cosine_similarity'] = max(
                chunk_data[chunk_id]['cosine_similarity'], 
                cos_score
            )
    
    # Add graph expansion results
    for chunk_id, g_score in graph_scores.items():
        if chunk_id not in chunk_data:
            # Node exists in graph but not in vector results
            # Retrieve text from graph database
            node = graph_db.get_node(chunk_id)
            chunk_text = node.text if node else ""
            
            chunk_data[chunk_id] = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'cosine_similarity': 0.0,
                'graph_score': g_score
            }
        else:
            chunk_data[chunk_id]['graph_score'] = g_score
    
    # Compute final scores
    final_results = []
    for chunk_id, data in chunk_data.items():
        final_score = (
            data['cosine_similarity'] * COSINE_WEIGHT + 
            data['graph_score'] * GRAPH_WEIGHT
        )
        
        result = {
            'chunk_id': chunk_id,
            'cosine_similarity': round(data['cosine_similarity'], 4),
            'graph_score': round(data['graph_score'], 2),
            'final_score': round(final_score, 4),
            'text': data['text']
        }
        final_results.append(result)
    
    # Sort by final score (descending)
    final_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return final_results


# ==================== 4. Final Retrieval Pipeline ====================

def hybrid_retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    End-to-end hybrid retrieval pipeline.
    
    Process:
        1. Perform vector search to get top_k chunks
        2. For each chunk, expand via graph to find related nodes
        3. Combine scores using hybrid ranking
        4. Return sorted results
    
    Args:
        query: User query string
        top_k: Number of initial vector search results
        
    Returns:
        JSON-serializable list of result dictionaries
    """
    # Step 1: Vector search
    vector_results = vector_search(query, top_k)
    
    # Step 2: Graph expansion
    graph_scores_combined = {}
    
    for chunk_id, _, cos_score in vector_results:
        # Get graph scores for this chunk
        related_scores = graph_score(graph_db.graph, chunk_id, depth=GRAPH_DEPTH)
        
        # Merge scores (take max if chunk appears in multiple expansions)
        for node_id, score in related_scores.items():
            if node_id not in graph_scores_combined:
                graph_scores_combined[node_id] = score
            else:
                graph_scores_combined[node_id] = max(
                    graph_scores_combined[node_id], 
                    score
                )
    
    # Step 3: Hybrid ranking
    final_results = hybrid_rank(vector_results, graph_scores_combined)
    
    return final_results


# ==================== 5. Example Usage ====================

if __name__ == "__main__":
    # Example query
    test_query = "How did Juliet die? Who killed Juliet?"
    
    print(f"\n{'='*60}")
    print(f"HYBRID RETRIEVAL SYSTEM")
    print(f"{'='*60}")
    print(f"\nQuery: {test_query}")
    print(f"\nConfiguration:")
    print(f"  - Cosine Weight: {COSINE_WEIGHT}")
    print(f"  - Graph Weight: {GRAPH_WEIGHT}")
    print(f"  - Graph Depth: {GRAPH_DEPTH}")
    print(f"\n{'='*60}\n")
    
    # Perform hybrid retrieval
    results = hybrid_retrieve(test_query, top_k=5)
    
    # Output as JSON
    print("Results (JSON):")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Also print human-readable format
    print(f"\n{'='*60}")
    print(f"Top {len(results)} Results:")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Chunk ID: {result['chunk_id']}")
        print(f"   Cosine Similarity: {result['cosine_similarity']}")
        print(f"   Graph Score: {result['graph_score']}")
        print(f"   Final Score: {result['final_score']}")
        print(f"   Text: {result['text'][:100]}...")
        print()
