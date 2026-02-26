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
        print(f"\n{'─'*70}")
        print(f"Query {i}: {query}")
        print(f"{'─'*70}\n")
        
        # Retrieve results
        results = hybrid_retrieve(query, top_k=5)
        
        # Display results
        if not results:
            print("No results found.")
            continue
        
        print(f"Found {len(results)} results:\n")
        
        for rank, result in enumerate(results[:10], 1):
            print(f"{rank}. [Score: {result['final_score']:.4f}]")
            print(f"   Chunk ID: {result['chunk_id']}")
            print(f"   Cosine: {result['cosine_similarity']:.4f} | Graph: {result['graph_score']:.2f}")
            print(f"   Text: {result['text'][:150]}...")
            print()
        
        # Save to JSON
        output_file = f"results_query_{i}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved full results to: {output_file}\n")
    
    print("="*70)
    print("Hybrid retrieval complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
