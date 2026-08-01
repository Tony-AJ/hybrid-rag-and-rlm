"""
DEBUG: Full Hybrid Retrieval Scoring Pipeline Diagnostic

Traces every step of the scoring pipeline to identify exactly where
vector scores might become 0.0 or otherwise incorrect.

Pipeline traced:
  Query -> Embedding -> Chroma Search -> Raw Distance -> Distance->Similarity
  -> Normalization -> Hybrid Score
"""

import os
import sys
import json
import numpy as np

# ── Suppress noisy warnings ──
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

import logging
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('posthog').setLevel(logging.ERROR)

# ── Imports ──
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from graph_db.graph_db import GraphDatabase

# ── Configuration (same as hybrid_retrieval.py) ──
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(CURRENT_DIR, "vector_db", "db", "chroma_db_with_metadata")
GRAPH_DB_PATH = os.path.join(CURRENT_DIR, "db", "graph_data.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COSINE_WEIGHT = 0.7   # alpha
GRAPH_WEIGHT = 0.3    # beta
PATH_PENALTY = 0.1    # gamma
GRAPH_DEPTH = 2
NORM_EPSILON = 1e-6

# ── Separator helpers ──
def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def section(title):
    print(f"\n--- {title} ---")

def kv(key, value):
    print(f"  {key}: {value}")


def main():
    query = "How did the feud end?"
    top_k = 5

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])

    header("HYBRID RETRIEVAL SCORING PIPELINE DEBUG")
    kv("Query", query)
    kv("top_k", top_k)
    kv("Vector DB Path", VECTOR_DB_PATH)
    kv("Graph DB Path", GRAPH_DB_PATH)

    # ================================================================
    # STEP 0: Verify databases exist
    # ================================================================
    section("STEP 0: Database Existence Check")
    kv("Vector DB exists", os.path.exists(VECTOR_DB_PATH))
    kv("Graph DB exists", os.path.exists(GRAPH_DB_PATH))

    if not os.path.exists(VECTOR_DB_PATH):
        print("\n  [FATAL] Vector DB directory missing! Run ingest_data.py first.")
        return
    if not os.path.exists(GRAPH_DB_PATH):
        print("\n  [FATAL] Graph DB file missing! Run ingest_data.py first.")
        return

    # ================================================================
    # STEP 1: Initialize Embeddings & Check dimensions
    # ================================================================
    section("STEP 1: Embedding Model Initialization")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Embed the query
    query_embedding = embeddings.embed_query(query)
    query_embedding_np = np.array(query_embedding)
    kv("Query embedding type", type(query_embedding).__name__)
    kv("Query embedding length", len(query_embedding))
    kv("Query embedding shape (np)", query_embedding_np.shape)
    kv("Query embedding norm (L2)", round(float(np.linalg.norm(query_embedding_np)), 6))
    kv("Query embedding first 5 values", [round(v, 6) for v in query_embedding[:5]])
    kv("Query embedding min", round(float(query_embedding_np.min()), 6))
    kv("Query embedding max", round(float(query_embedding_np.max()), 6))

    # ================================================================
    # STEP 2: Open ChromaDB & verify stored documents
    # ================================================================
    section("STEP 2: ChromaDB Contents Inspection")
    chroma_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    # Access underlying Chroma collection
    collection = chroma_db._collection
    count = collection.count()
    kv("Total documents in ChromaDB", count)

    if count == 0:
        print("\n  [FATAL] ChromaDB is empty! Run ingest_data.py first.")
        return

    # Peek at stored documents
    peek = collection.peek(limit=3)
    print("\n  Peeked documents (first 3):")
    for i in range(min(3, len(peek['ids']))):
        doc_id = peek['ids'][i]
        doc_text = peek['documents'][i][:80] + "..." if peek['documents'][i] and len(peek['documents'][i]) > 80 else peek['documents'][i]
        doc_meta = peek['metadatas'][i] if peek['metadatas'] else {}
        has_embedding = peek['embeddings'] is not None and len(peek['embeddings']) > i and peek['embeddings'][i] is not None
        emb_dim = len(peek['embeddings'][i]) if has_embedding else "MISSING"
        print(f"    [{i}] ID={doc_id}")
        print(f"        Text: {doc_text}")
        print(f"        Metadata: {doc_meta}")
        print(f"        Embedding dim: {emb_dim}")

    # ================================================================
    # STEP 3: Verify one specific document embedding
    # ================================================================
    section("STEP 3: Document Embedding Verification (chunk_1)")
    try:
        doc_result = collection.get(ids=["chunk_1"], include=["embeddings", "documents", "metadatas"])
        if doc_result and doc_result['ids']:
            kv("chunk_1 text exists", bool(doc_result['documents'][0]))
            kv("chunk_1 text (first 80 chars)", doc_result['documents'][0][:80])
            has_emb = doc_result['embeddings'] is not None and len(doc_result['embeddings']) > 0 and doc_result['embeddings'][0] is not None
            kv("chunk_1 embedding exists", has_emb)
            if has_emb:
                emb = np.array(doc_result['embeddings'][0])
                kv("chunk_1 embedding dim", emb.shape)
                kv("chunk_1 embedding norm (L2)", round(float(np.linalg.norm(emb)), 6))
                kv("Dimension match with query?", emb.shape[0] == query_embedding_np.shape[0])
            else:
                print("  [BUG] Embedding is MISSING for chunk_1!")
        else:
            print("  [WARNING] chunk_1 not found in ChromaDB")
    except Exception as e:
        print(f"  [ERROR] Could not fetch chunk_1: {e}")

    # ================================================================
    # STEP 4: Raw ChromaDB search (distances)
    # ================================================================
    section("STEP 4: Raw ChromaDB Search Output")

    # Use the underlying collection.query for full control
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    raw_ids = raw_results['ids'][0]
    raw_docs = raw_results['documents'][0]
    raw_distances = raw_results['distances'][0]
    raw_metas = raw_results['metadatas'][0]

    print(f"\n  Raw results from ChromaDB (n={len(raw_ids)}):")
    print(f"  Distance metric: (ChromaDB default = L2 squared distance)")
    print()

    for i in range(len(raw_ids)):
        doc_id = raw_ids[i]
        distance = raw_distances[i]
        text_preview = raw_docs[i][:80] + "..." if len(raw_docs[i]) > 80 else raw_docs[i]
        print(f"    [{i+1}] ID={doc_id}")
        print(f"        Raw Distance: {distance}")
        print(f"        Text: {text_preview}")
        print(f"        Metadata: {raw_metas[i]}")

    # ================================================================
    # STEP 5: Distance -> Similarity Conversion
    # ================================================================
    section("STEP 5: Distance -> Similarity Conversion")
    print()
    print("  Pipeline v2.0 method: cosine = max(0, 1.0 - (L2_squared / 2))")
    print("  Exact for unit-normalized embeddings (all-MiniLM-L6-v2 is normalized).")
    print("  Legacy v1.0 method (shown for comparison only): 1.0 / (1.0 + distance)")
    print()

    similarities_current = []
    similarities_legacy = []

    for i in range(len(raw_ids)):
        d = raw_distances[i]
        # v2.0 conversion: L2 squared -> true cosine similarity
        sim_current = max(0.0, 1.0 - (d / 2.0))
        similarities_current.append(sim_current)
        # v1.0 conversion, kept to show the range compression it caused
        sim_legacy = 1.0 / (1.0 + d)
        similarities_legacy.append(sim_legacy)

        print(f"    [{raw_ids[i]}] distance={d:.6f}  ->  1-(d/2)={sim_current:.6f}  |  legacy 1/(1+d)={sim_legacy:.6f}")

    # ================================================================
    # STEP 5a: LangChain similarity_search_with_score comparison
    # ================================================================
    section("STEP 5a: LangChain similarity_search_with_score Output")
    lc_results = chroma_db.similarity_search_with_score(query, k=top_k)

    print(f"\n  LangChain returns {len(lc_results)} results.")
    print("  Note: LangChain Chroma returns the RAW DISTANCE as the 'score'.")
    print()

    for i, (doc, score) in enumerate(lc_results):
        chunk_id = doc.metadata.get('id', doc.metadata.get('source', 'unknown'))
        sim = max(0.0, 1.0 - (score / 2.0))
        print(f"    [{i+1}] ID={chunk_id}  LangChain score(=distance)={score:.6f}  -> cosine={sim:.6f}")
        print(f"         Text: {doc.page_content[:80]}...")

    # ================================================================
    # STEP 6: Min-Max Normalization Analysis
    # ================================================================
    section("STEP 6: Min-Max Normalization of Vector Scores")

    raw_sims = similarities_current  # v2.0 conversion
    s_min = min(raw_sims) if raw_sims else 0.0
    s_max = max(raw_sims) if raw_sims else 1.0
    s_range = s_max - s_min

    kv("Raw cosine scores", [round(s, 6) for s in raw_sims])
    kv("min(cosine)", round(s_min, 6))
    kv("max(cosine)", round(s_max, 6))
    kv("range (max - min)", round(s_range, 6))
    print()
    print(f"  v2.0 formula: normalized = (score - min) / (range + {NORM_EPSILON})")
    print("  No floor clamp. Worst vector hit normalizes to 0.0 but keeps its raw cosine;")
    print("  nodes never retrieved by vector get raw=null, normalized=null instead.")

    if s_range == 0:
        print("\n  [NOTE] All cosine scores are IDENTICAL.")
        print(f"  -> range=0, so epsilon dominates: every normalized value collapses to ~0.0")
        print("  -> Ranking then falls back entirely to the graph component.")

    normalized = []
    for i, sim in enumerate(raw_sims):
        norm = (sim - s_min) / (s_range + NORM_EPSILON)
        normalized.append(norm)
        print(f"    [{raw_ids[i]}] raw_cosine={sim:.6f} -> normalized={norm:.6f}")

    # ================================================================
    # STEP 7: Graph Scores
    # ================================================================
    section("STEP 7: Graph Expansion & Scores")
    graph_db = GraphDatabase(db_path=GRAPH_DB_PATH, auto_persist=True)
    kv("Graph nodes", graph_db.graph.number_of_nodes())
    kv("Graph edges", graph_db.graph.number_of_edges())

    # Check which vector-hit nodes exist in graph
    print("\n  Vector hit nodes in graph?")
    for doc_id in raw_ids:
        in_graph = doc_id in graph_db.graph.nodes
        neighbors = []
        if in_graph:
            neighbors = list(set(graph_db.graph.successors(doc_id)) | set(graph_db.graph.predecessors(doc_id)))
        print(f"    {doc_id}: in_graph={in_graph}, neighbors={neighbors}")

    # Compute graph scores for top vector hit
    top_vector_id = raw_ids[0]
    print(f"\n  Graph expansion from top vector hit: {top_vector_id}")

    from hybrid_retrieval import graph_score
    g_scores = graph_score(graph_db.graph, top_vector_id, depth=GRAPH_DEPTH)
    for nid, info in g_scores.items():
        print(f"    {nid}: score={info['score']:.4f}, hop={info['hop']}, via={info['via_edge']}, from={info['from_node']}")

    # ================================================================
    # STEP 8: Final Hybrid Score Computation
    # ================================================================
    section("STEP 8: Final Hybrid Score Computation")
    alpha = COSINE_WEIGHT
    beta = GRAPH_WEIGHT
    gamma = PATH_PENALTY

    kv("alpha (vector weight)", alpha)
    kv("beta (graph weight)", beta)
    kv("gamma (path penalty)", gamma)
    print()

    # Combine all candidates (vector + graph expansion)
    all_candidates = {}

    print("  Split formula:")
    print("    vector-retrieved: final = alpha*norm_v + beta*g - gamma*penalty")
    print("    graph-only:       final = beta*g - gamma*penalty   (no fake vector term)")
    print()

    # Add vector results
    for i, doc_id in enumerate(raw_ids):
        all_candidates[doc_id] = {
            "norm_v": normalized[i],
            "raw_sim": raw_sims[i],
            "retrieved_by_vector": True,
            "g_score": g_scores.get(doc_id, {}).get("score", 0.0),
            "hop": g_scores.get(doc_id, {}).get("hop", 0),
        }

    # Add graph-only results (raw/normalized stay null, not 0.0)
    for nid, info in g_scores.items():
        if nid not in all_candidates:
            all_candidates[nid] = {
                "norm_v": None,
                "raw_sim": None,
                "retrieved_by_vector": False,
                "g_score": info["score"],
                "hop": info["hop"],
            }

    print(f"  {'ID':<12} {'raw_cos':>10} {'norm_v':>10} {'g_score':>10} {'hop':>5} {'penalty':>10} {'final':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*5} {'-'*10} {'-'*10}")

    final_results = []
    for nid, data in all_candidates.items():
        norm_g = data["g_score"]
        hop = data["hop"]
        penalty = (hop / GRAPH_DEPTH) if GRAPH_DEPTH > 0 else 0.0

        if data["retrieved_by_vector"]:
            norm_v = data["norm_v"]
            final = (alpha * norm_v) + (beta * norm_g) - (gamma * penalty)
        else:
            norm_v = None
            final = (beta * norm_g) - (gamma * penalty)
        final = max(0.0, final)

        raw_str = f"{data['raw_sim']:.6f}" if data["raw_sim"] is not None else "null"
        norm_str = f"{norm_v:.6f}" if norm_v is not None else "null"
        print(f"  {nid:<12} {raw_str:>10} {norm_str:>10} {norm_g:>10.6f} {hop:>5} {penalty:>10.4f} {final:>10.4f}")
        final_results.append((nid, final, norm_v, norm_g, hop))

    final_results.sort(key=lambda x: x[1], reverse=True)

    # ================================================================
    # STEP 9: Diagnosis Summary
    # ================================================================
    header("DIAGNOSIS SUMMARY")

    issues_found = []

    # Check 1: Are distances reasonable?
    if all(d < 0.001 for d in raw_distances):
        issues_found.append("[ISSUE] All distances are near-zero. Possible duplicate embeddings or query matches stored text exactly.")

    # Check 2: Are all similarities identical?
    if s_range == 0 and len(raw_sims) > 1:
        issues_found.append("[ISSUE] All cosine scores are identical. Normalization collapses to ~0.0 "
                          "and ranking is decided entirely by the graph component.")

    # Check 3: Is the cosine range too narrow for min-max to be meaningful?
    if s_max > 0 and (s_max - s_min) / s_max < 0.05 and len(raw_sims) > 1:
        issues_found.append(f"[ISSUE] Cosine range is very narrow ({s_min:.6f} to {s_max:.6f}). "
                          f"Min-max normalization will amplify tiny differences into large "
                          f"normalized gaps. Inspect raw_vector_score, not the normalized one.")

    # Check 4: Are embeddings present?
    if count == 0:
        issues_found.append("[ISSUE] No documents in ChromaDB. Run ingest_data.py.")

    # Check 5: Dimension mismatch
    try:
        doc_result = collection.get(ids=[raw_ids[0]], include=["embeddings"])
        if doc_result['embeddings'] and doc_result['embeddings'][0]:
            doc_dim = len(doc_result['embeddings'][0])
            q_dim = len(query_embedding)
            if doc_dim != q_dim:
                issues_found.append(f"[ISSUE] Dimension mismatch: query={q_dim}, document={doc_dim}")
    except Exception:
        pass

    # Check 6: Graph nodes missing
    missing_in_graph = [nid for nid in raw_ids if nid not in graph_db.graph.nodes]
    if missing_in_graph:
        issues_found.append(f"[ISSUE] Vector-hit nodes missing from graph: {missing_in_graph}")

    if issues_found:
        print("\n  Issues detected:")
        for issue in issues_found:
            print(f"    {issue}")
    else:
        print("\n  No obvious issues detected in the pipeline.")

    print(f"\n  Final ranked results:")
    for rank, (nid, final, nv, ng, hop) in enumerate(final_results, 1):
        nv_str = f"{nv:.4f}" if nv is not None else "null"
        print(f"    {rank}. {nid}: final={final:.4f} (vector={nv_str}, graph={ng:.4f}, hop={hop})")

    print()


if __name__ == "__main__":
    main()
