import os
from typing import List, Dict, Any, Optional
import collections
import shutil
from datetime import datetime, timezone
from fastapi import UploadFile
from pypdf import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.models import (
    NodeCreate, NodeResponse, NodeCreateResponse, NodeUpdate, NodeUpdateResponse, NodeDeleteResponse,
    EdgeCreate, EdgeResponse, EdgeCreateResponse, EdgeGetResponse, EdgeUpdate, EdgeUpdateResponse, EdgeDeleteResponse,
    VectorSearchResultItem, VectorSearchResponse,
    GraphTraversalResponse, GraphTraversalNode,
    HybridSearchResultItem, HybridSearchResponse, RetrievalReason, RetrievalMetadata
)
from app.vector_db import VectorDatabase
from graph_db.graph_db import GraphDatabase
from graph_db.models import GraphNode, GraphRelationship

# Normalization epsilon (prevents division by zero)
NORM_EPSILON = 1e-6

# Pipeline version (bump on any scoring-formula change)
PIPELINE_VERSION = "2.0"

# Candidate pool for the PDF upload flow, which surfaces only the single best chunk.
# Normalization needs more than one sample to mean anything.
PDF_CANDIDATE_POOL = 5

class HybridRetrievalService:
    def __init__(self, vector_db_path: Optional[str] = None, graph_db_path: Optional[str] = None):
        # Paths
        current_dir = os.getcwd()
        self.vector_db_path = vector_db_path or os.getenv("VECTOR_DB_PATH") or os.path.join(current_dir, "db", "chroma_db_service")
        self.graph_db_path = graph_db_path or os.getenv("GRAPH_DB_PATH") or os.path.join(current_dir, "db", "graph_data_service.json")
        self.books_dir = os.path.join(current_dir, "vector_db", "books")
        os.makedirs(self.books_dir, exist_ok=True)
        
        # Initialize DBs
        self.vector_db = VectorDatabase(persist_directory=self.vector_db_path)
        self.graph_db = GraphDatabase(db_path=self.graph_db_path, auto_persist=True)

    # ==================== Node Operations ====================

    def create_node(self, node_data: NodeCreate) -> NodeCreateResponse:
        # Create in GraphDB
        graph_node = self.graph_db.create_node(
            node_id=node_data.id,
            text=node_data.text,
            metadata=node_data.metadata,
            embedding=node_data.embedding
        )
        
        embedding_dim = None
        if node_data.regen_embedding:
            # Add to VectorDB (which generates embedding)
            # We assume model dim is 384 for all-MiniLM-L6-v2
            embedding_dim = 384
            
            # Add 'id' to metadata for VectorDB retrieval
            meta = node_data.metadata.copy()
            meta['id'] = node_data.id
            
            # Sanitize metadata for Chroma (no lists allowed)
            chroma_meta = {}
            for k, v in meta.items():
                if isinstance(v, list):
                    chroma_meta[k] = ", ".join(map(str, v))
                else:
                    chroma_meta[k] = v
            
            self.vector_db.add_document(node_data.id, node_data.text, chroma_meta)
            
        return NodeCreateResponse(
            status="created",
            id=node_data.id,
            embedding_dim=embedding_dim
        )

    def get_node(self, node_id: str) -> Optional[NodeResponse]:
        graph_node = self.graph_db.get_node(node_id)
        if not graph_node:
            return None
            
        # Get embedding from VectorDB if possible, or GraphDB
        embedding = graph_node.embedding
        if not embedding:
            try:
                result = self.vector_db.db.get(ids=[node_id], include=['embeddings'])
                # Check if embeddings are present and not empty
                if result and 'embeddings' in result and result['embeddings'] is not None and len(result['embeddings']) > 0:
                    embedding = result['embeddings'][0]
            except Exception as e:
                # print(f"Error fetching embedding: {e}")
                pass
        
        # Get edges
        edges = []
        if node_id in self.graph_db.graph:
            # Outgoing edges
            for neighbor in self.graph_db.graph.successors(node_id):
                for key, edge_data in self.graph_db.graph[node_id][neighbor].items():
                    edges.append({
                        "edge_id": edge_data.get("id"),
                        "target": neighbor,
                        "type": edge_data.get("type"),
                        "weight": edge_data.get("weight")
                    })
        
        return NodeResponse(
            id=graph_node.id,
            text=graph_node.text,
            metadata=graph_node.metadata,
            embedding=embedding,
            edges=edges
        )

    def update_node(self, node_id: str, update_data: NodeUpdate) -> Optional[NodeUpdateResponse]:
        # Check if node exists
        if not self.graph_db.get_node(node_id):
            return None
            
        # Update GraphDB
        self.graph_db.update_node(
            node_id=node_id,
            text=update_data.text,
            metadata=update_data.metadata
        )
        
        embedding_regenerated = False
        if update_data.regen_embedding:
            # Get current text (updated or existing)
            current_node = self.graph_db.get_node(node_id)
            if current_node:
                meta = current_node.metadata.copy()
                meta['id'] = node_id
                
                # Sanitize metadata for Chroma
                chroma_meta = {}
                for k, v in meta.items():
                    if isinstance(v, list):
                        chroma_meta[k] = ", ".join(map(str, v))
                    else:
                        chroma_meta[k] = v
                        
                self.vector_db.update_document(node_id, current_node.text, chroma_meta)
                embedding_regenerated = True
                
        return NodeUpdateResponse(
            status="updated",
            id=node_id,
            embedding_regenerated=embedding_regenerated
        )

    def delete_node(self, node_id: str) -> Optional[NodeDeleteResponse]:
        # Count edges to be removed
        removed_edges_count = 0
        if node_id in self.graph_db.graph:
            removed_edges_count += self.graph_db.graph.out_degree(node_id)
            removed_edges_count += self.graph_db.graph.in_degree(node_id)
            
        success = self.graph_db.delete_node(node_id)
        if not success:
            return None
            
        # Delete from VectorDB
        self.vector_db.delete_document(node_id)
        
        return NodeDeleteResponse(
            status="deleted",
            id=node_id,
            removed_edges_count=removed_edges_count
        )

    # ==================== Edge Operations ====================

    def create_edge(self, edge_data: EdgeCreate) -> Optional[EdgeCreateResponse]:
        edge = self.graph_db.create_edge(
            source_id=edge_data.source,
            target_id=edge_data.target,
            rel_type=edge_data.type,
            weight=edge_data.weight
        )
        if not edge:
            return None
            
        return EdgeCreateResponse(
            status="created",
            edge_id=edge.id,
            source=edge.source,
            target=edge.target
        )

    def get_edge(self, edge_id: str) -> Optional[EdgeGetResponse]:
        edge = self.graph_db.get_edge(edge_id)
        if not edge:
            return None
            
        return EdgeGetResponse(
            edge_id=edge.id,
            source=edge.source,
            target=edge.target,
            type=edge.type,
            weight=edge.weight
        )

    def update_edge(self, edge_id: str, update_data: EdgeUpdate) -> Optional[EdgeUpdateResponse]:
        if edge_id not in self.graph_db._edge_id_map:
            return None
            
        source, target, key = self.graph_db._edge_id_map[edge_id]
        self.graph_db.graph[source][target][key]['weight'] = update_data.weight
        
        if self.graph_db.auto_persist:
            self.graph_db.save()
            
        return EdgeUpdateResponse(
            status="updated",
            edge_id=edge_id,
            new_weight=update_data.weight
        )

    def delete_edge(self, edge_id: str) -> Optional[EdgeDeleteResponse]:
        success = self.graph_db.delete_edge(edge_id)
        if not success:
            return None
            
        return EdgeDeleteResponse(
            status="deleted",
            edge_id=edge_id
        )

    # ==================== Search Operations ====================

    def vector_search(self, query: str, top_k: int, filter: Optional[Dict[str, Any]] = None) -> VectorSearchResponse:
        results = self.vector_db.search(query, top_k, filter=filter)
        items = []
        for doc_id, text, score, metadata in results:
            nid = doc_id or metadata.get('id')
            if not nid:
                continue
                
            items.append(VectorSearchResultItem(
                id=nid,
                vector_score=round(score, 4)
            ))
            
        return VectorSearchResponse(
            query_text=query,
            results=items
        )

    def graph_traversal(self, start_id: str, depth: int, type_filter: Optional[str] = None) -> Optional[GraphTraversalResponse]:
        if not self.graph_db.get_node(start_id):
            return None
            
        # BFS with path tracking
        visited = {start_id}
        queue = collections.deque([(start_id, 0, [], [])]) # node_id, depth, edge_types, edge_weights
        
        nodes = []
        
        while queue:
            curr_id, curr_depth, path_types, path_weights = queue.popleft()
            
            if curr_depth > 0:
                node_info = {
                    "id": curr_id,
                    "hop": curr_depth
                }
                
                if curr_depth == 1:
                    node_info["edge"] = path_types[0]
                    node_info["weight"] = path_weights[0]
                else:
                    node_info["edge_path"] = path_types
                    node_info["weights"] = path_weights
                    
                nodes.append(GraphTraversalNode(**node_info))
            
            if curr_depth < depth:
                if curr_id in self.graph_db.graph:
                    for neighbor in self.graph_db.graph.successors(curr_id):
                        if neighbor in visited:
                            continue
                            
                        # Check edges
                        # MultiDiGraph can have multiple edges. We take the first one or iterate?
                        # Prompt implies simple traversal.
                        # We'll take the first edge that matches filter (if any)
                        
                        edge_found = False
                        for key, edge_data in self.graph_db.graph[curr_id][neighbor].items():
                            etype = edge_data.get("type")
                            eweight = edge_data.get("weight")
                            
                            if type_filter and etype != type_filter:
                                continue
                                
                            visited.add(neighbor)
                            queue.append((
                                neighbor, 
                                curr_depth + 1, 
                                path_types + [etype], 
                                path_weights + [eweight]
                            ))
                            edge_found = True
                            break # Only follow one edge to a neighbor to avoid duplicates in simple BFS
                            
        return GraphTraversalResponse(
            start_id=start_id,
            depth=depth,
            nodes=nodes
        )

    def hybrid_search(self, query: str, vector_weight: float, graph_weight: float, top_k: int, max_depth: int = 2, path_distance_penalty: float = 0.1) -> HybridSearchResponse:
        """
        Research-grade Hybrid Retrieval with full explainability (v2.0).

        Design:
            - Vector-retrieved nodes: final = alpha * norm_v + beta * g - gamma * penalty
            - Graph-only nodes:       final = beta * g - gamma * penalty
            - No fake vector scores for graph-only nodes.

        Normalization:
            normalized = (score - v_min) / (v_range + epsilon)
            No floor clamp. retrieved_by_vector boolean distinguishes 0.0 from null.
        """
        # ── Step 1: Vector Search ──
        vector_results = self.vector_db.search(query, top_k)

        candidates = {}
        start_nodes = []

        for vector_rank, (doc_id, text, score, metadata) in enumerate(vector_results, 1):
            nid = doc_id or metadata.get('id')
            if not nid:
                continue

            candidates[nid] = {
                "raw_vector_score": score,         # true cosine similarity
                "retrieved_by_vector": True,
                "vector_rank": vector_rank,
                "graph_score": 0.0,
                "graph_rank": None,
                "hop": 0,
                "reached_via_edge": None,
                "reached_from": None,
                "text": text,
                "metadata": metadata,
            }
            start_nodes.append(nid)

        # ── Step 2: Graph Expansion ──
        queue = collections.deque()
        visited = {}
        graph_rank_counter = 0

        for nid in start_nodes:
            if nid not in visited:
                visited[nid] = 0
                queue.append((nid, 0, None, None))

        while queue:
            curr_id, hops, via_edge, from_node = queue.popleft()

            g_score = 1.0 / (1.0 + hops)
            graph_rank_counter += 1

            if curr_id not in candidates:
                # Graph-only node
                node = self.graph_db.get_node(curr_id)
                node_text = node.text if node else ""
                node_meta = node.metadata if node else {}

                candidates[curr_id] = {
                    "raw_vector_score": None,       # null, not 0.0
                    "retrieved_by_vector": False,
                    "vector_rank": None,
                    "graph_score": g_score,
                    "graph_rank": graph_rank_counter,
                    "hop": hops,
                    "reached_via_edge": via_edge,
                    "reached_from": from_node,
                    "text": node_text,
                    "metadata": node_meta,
                }
            else:
                if g_score > candidates[curr_id]["graph_score"]:
                    candidates[curr_id]["graph_score"] = g_score
                    candidates[curr_id]["hop"] = hops
                    candidates[curr_id]["reached_via_edge"] = via_edge
                    candidates[curr_id]["reached_from"] = from_node
                    if candidates[curr_id]["graph_rank"] is None:
                        candidates[curr_id]["graph_rank"] = graph_rank_counter

            if hops < max_depth:
                if curr_id in self.graph_db.graph:
                    for neighbor in self.graph_db.graph.successors(curr_id):
                        if neighbor not in visited:
                            edge_type = None
                            for key, edge_data in self.graph_db.graph[curr_id][neighbor].items():
                                edge_type = edge_data.get("type", "unknown")
                                break
                            visited[neighbor] = hops + 1
                            queue.append((neighbor, hops + 1, edge_type, curr_id))

        # ── Step 3: Normalize Vector Scores (only vector-retrieved) ──
        raw_scores = [d["raw_vector_score"] for d in candidates.values() if d["retrieved_by_vector"]]
        v_min = min(raw_scores) if raw_scores else 0.0
        v_max = max(raw_scores) if raw_scores else 0.0
        v_range = v_max - v_min

        # Min-max is undefined with a single hit or with all scores tied: (score - min) is
        # 0 for every candidate, which would silently zero out the entire vector component.
        # There is no relative ordering to express, so every hit is treated as best-available.
        degenerate = len(raw_scores) < 2 or v_range < NORM_EPSILON

        for data in candidates.values():
            if not data["retrieved_by_vector"]:
                data["norm_vector_score"] = None  # null, not 0.0
            elif degenerate:
                data["norm_vector_score"] = 1.0
            else:
                data["norm_vector_score"] = (data["raw_vector_score"] - v_min) / (v_range + NORM_EPSILON)

        # ── Step 4: Final Scoring with Split Formula ──
        alpha = vector_weight
        beta = graph_weight
        gamma = path_distance_penalty

        results = []
        for nid, data in candidates.items():
            norm_g = data["graph_score"]
            hop = data["hop"]
            penalty = (hop / max_depth) if max_depth > 0 else 0.0

            # Split formula
            if data["retrieved_by_vector"]:
                norm_v = data["norm_vector_score"]
                vector_contrib = alpha * norm_v
                graph_contrib = beta * norm_g
                penalty_contrib = gamma * penalty
                final = vector_contrib + graph_contrib - penalty_contrib
            else:
                vector_contrib = 0.0
                graph_contrib = beta * norm_g
                penalty_contrib = gamma * penalty
                final = graph_contrib - penalty_contrib

            # Keep the pre-clamp value so the contribution breakdown always reconstructs
            # exactly; a heavy penalty can drive a deep graph-only node below zero.
            unclamped_final = final
            final = max(0.0, final)

            # ── Build retrieval_sources ──
            sources = []
            if data["retrieved_by_vector"]:
                sources.append("vector")
            if data["graph_score"] > 0:
                sources.append("graph")

            # ── Build structured reason ──
            reason = RetrievalReason(
                vector_rank=data["vector_rank"],
                graph_rank=data["graph_rank"],
                hop=hop,
                edge=data.get("reached_via_edge"),
                retrieval_method=sources,
            )

            results.append(HybridSearchResultItem(
                node_id=nid,
                raw_vector_score=round(data["raw_vector_score"], 4) if data["raw_vector_score"] is not None else None,
                normalized_vector_score=round(data["norm_vector_score"], 4) if data["norm_vector_score"] is not None else None,
                graph_score=round(norm_g, 4),
                vector_contribution=round(vector_contrib, 4),
                graph_contribution=round(graph_contrib, 4),
                penalty_contribution=round(penalty_contrib, 4),
                unclamped_final_score=round(unclamped_final, 4),
                final_score=round(final, 4),
                retrieval_sources=sources,
                reason=reason,
                metadata=data.get("metadata", {}),
                text=data.get("text", ""),
                info={},
            ))

        results.sort(key=lambda x: x.final_score, reverse=True)

        # Assign overall fused rank (vector_rank / graph_rank stay as retrieved)
        for rank, r in enumerate(results, 1):
            r.reason.rank = rank

        return HybridSearchResponse(
            query_text=query,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            pipeline_version=PIPELINE_VERSION,
            retrieval_metadata=RetrievalMetadata(
                pipeline_version=PIPELINE_VERSION,
                timestamp=datetime.now(timezone.utc).isoformat(),
                max_depth=max_depth,
                path_distance_penalty=path_distance_penalty,
            ),
            results=results[:top_k]
        )

    # ==================== PDF Operations ====================

    def process_pdf_and_search(self, file: UploadFile, query: str) -> Optional[Any]:
        # 1. Save File
        file_path = os.path.join(self.books_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # 2. Extract Text
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        # 3. Chunk Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=40
        )
        chunks = text_splitter.split_text(text)
        
        # 4. Create Nodes & Edges
        prev_id = None
        created_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file.filename}_chunk_{i}"
            created_ids.append(chunk_id)
            
            # Create Node
            self.create_node(NodeCreate(
                id=chunk_id,
                text=chunk,
                metadata={"source": file.filename, "chunk_index": i},
                regen_embedding=True
            ))
            
            # Create Edge to previous
            if prev_id:
                self.create_edge(EdgeCreate(
                    source=prev_id,
                    target=chunk_id,
                    type="next_chunk",
                    weight=1.0
                ))
            prev_id = chunk_id
            
        # 5. Hybrid Search
        # Retrieve a real candidate pool even though only the top hit is returned:
        # at top_k=1 min-max normalization has a single sample and carries no signal.
        search_res = self.hybrid_search(query, vector_weight=0.5, graph_weight=0.5, top_k=PDF_CANDIDATE_POOL)
        
        if not search_res.results:
            return None
            
        top_result = search_res.results[0]
        
        # Get text for the result
        node = self.get_node(top_result.node_id)
        text_content = node.text if node else top_result.text
        
        # Return in the format expected by test_pdf_flow.py (HybridSearchResult)
        from app.models import HybridSearchResult
        return HybridSearchResult(
            node_id=top_result.node_id,
            final_score=top_result.final_score,
            cosine_similarity=top_result.raw_vector_score or 0.0,
            graph_score=top_result.graph_score,
            text=text_content
        )
