"""
Main entry point for Hybrid Retrieval System

Demonstrates end-to-end hybrid retrieval combining:
- ChromaDB vector search (cosine similarity)
- NetworkX graph relations (graph-based scoring)
"""

from hybrid_retrieval import hybrid_retrieve
import json


def main():
    """
    Execute hybrid retrieval system with example queries.
    """
    # Example queries
    queries = [
        "How did the feud end?",
        "Who killed Tybalt?",
        "Why did Juliet take the potion?"
    ]
    
    print("\n" + "="*70)
    print("HYBRID RETRIEVAL SYSTEM - ChromaDB + NetworkX")
    print("="*70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'-'*70}")
        print(f"Query {i}: {query}")
        print(f"{'-'*70}\n")
        
        # Retrieve results (envelope carries retrieval_metadata for reproducibility)
        envelope = hybrid_retrieve(query, top_k=5, with_metadata=True)
        results = envelope["results"]

        # Display results
        if not results:
            print("No results found.")
            continue

        print(f"Found {len(results)} results:\n")

        for result in results[:10]:
            reason = result["reason"]
            raw_v = result["raw_vector_score"]
            norm_v = result["normalized_vector_score"]
            raw_v_str = f"{raw_v:.4f}" if raw_v is not None else "N/A"
            norm_v_str = f"{norm_v:.4f}" if norm_v is not None else "N/A"

            print(f"{reason['rank']}. [Score: {result['final_score']:.4f}] via {', '.join(result['retrieval_sources'])}")
            print(f"   Node ID: {result['node_id']}")
            print(f"   Cosine: {raw_v_str} | Normalized: {norm_v_str} | Graph: {result['graph_score']:.4f}")
            print(f"   Contributions: vector={result['vector_contribution']:.4f} "
                  f"graph={result['graph_contribution']:.4f} penalty={result['penalty_contribution']:.4f}")
            print(f"   Reason: hop={reason['hop']}, edge={reason['edge']}, "
                  f"vector_rank={reason['vector_rank']}, graph_rank={reason['graph_rank']}")
            print(f"   Text: {result['text'][:150]}...")
            print()

        # Save to JSON
        output_file = f"results_query_{i}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(envelope, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved full results to: {output_file}\n")
    
    print("="*70)
    print("Hybrid retrieval complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
