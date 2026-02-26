import os
from typing import List, Dict, Any, Optional
import collections
import shutil
from fastapi import UploadFile
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.models import (
    NodeCreate, NodeResponse, NodeCreateResponse, NodeUpdate, NodeUpdateResponse, NodeDeleteResponse,
    EdgeCreate, EdgeResponse, EdgeCreateResponse, EdgeGetResponse, EdgeUpdate, EdgeUpdateResponse, EdgeDeleteResponse,
    VectorSearchResultItem, VectorSearchResponse,
    GraphTraversalResponse, GraphTraversalNode,
    HybridSearchResultItem, HybridSearchResponse
)
from app.vector_db import VectorDatabase
from graph_db.graph_db import GraphDatabase
from graph_db.models import GraphNode, GraphRelationship

class HybridRetrievalService:
    def __init__(self):
        # Paths
        current_dir = os.getcwd()
        self.vector_db_path = os.path.join(current_dir, "db", "chroma_db_service")
        self.graph_db_path = os.path.join(current_dir, "db", "graph_data_service.json")
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

    def hybrid_search(self, query: str, vector_weight: float, graph_weight: float, top_k: int) -> HybridSearchResponse:
        # 1. Vector Search
        vector_results = self.vector_db.search(query, top_k)
        
        candidates = {}
        start_nodes = []
        
        for doc_id, text, score, metadata in vector_results:
            nid = doc_id or metadata.get('id')
            if not nid: continue
            
            candidates[nid] = {
                "vector_score": score,
                "graph_score": 0.0,
                "info": {"hop": 0}
            }
            start_nodes.append(nid)
            
        # 2. Graph Scoring: 1 / (1 + hops)
        queue = collections.deque()
        visited = {} 
        
        for nid in start_nodes:
            if nid not in visited:
                visited[nid] = 0
                queue.append((nid, 0))
                
        while queue:
            curr_id, hops = queue.popleft()
            
            g_score = 1.0 / (1.0 + hops)
            
            if curr_id not in candidates:
                candidates[curr_id] = {
                    "vector_score": 0.0,
                    "graph_score": g_score,
                    "info": {"hop": hops}
                }
            else:
                candidates[curr_id]["graph_score"] = g_score
                candidates[curr_id]["info"]["hop"] = hops
                
            if hops < 2:
                if curr_id in self.graph_db.graph:
                     for neighbor in self.graph_db.graph.successors(curr_id):
                         if neighbor not in visited:
                             visited[neighbor] = hops + 1
                             queue.append((neighbor, hops + 1))

        # 3. Final Ranking
        results = []
        for nid, data in candidates.items():
            final = (data["vector_score"] * vector_weight) + (data["graph_score"] * graph_weight)
            
            results.append(HybridSearchResultItem(
                id=nid,
                vector_score=round(data["vector_score"], 4),
                graph_score=round(data["graph_score"], 4),
                final_score=round(final, 4),
                info=data["info"]
            ))
            
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return HybridSearchResponse(
            query_text=query,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
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
        # We use default weights from the prompt/requirement if not specified
        # But here we just want the top result for the PDF test
        search_res = self.hybrid_search(query, vector_weight=0.5, graph_weight=0.5, top_k=1)
        
        if not search_res.results:
            return None
            
        top_result = search_res.results[0]
        
        # Get text for the result
        node = self.get_node(top_result.id)
        text_content = node.text if node else ""
        
        # Return in the format expected by test_pdf_flow.py (HybridSearchResult)
        # {
        #   "node_id": str,
        #   "final_score": float,
        #   "cosine_similarity": float,
        #   "graph_score": float,
        #   "text": str
        # }
        from app.models import HybridSearchResult
        return HybridSearchResult(
            node_id=top_result.id,
            final_score=top_result.final_score,
            cosine_similarity=top_result.vector_score,
            graph_score=top_result.graph_score,
            text=text_content
        )
