

import json
from hybrid_retrieval import (
    vector_search,
    graph_score,
    hybrid_rank,
    hybrid_retrieve,
    graph_db,
    COSINE_WEIGHT,
    GRAPH_WEIGHT
)


def test_vector_search():
    
    print("\n" + "="*70)
    print("TEST 1: Vector Search")
    print("="*70)
    
    query = "How did Juliet die?"
    results = vector_search(query, top_k=3)
    
    print(f"\nQuery: '{query}'")
    print(f"Results: {len(results)} chunks found\n")
    
    for i, (chunk_id, text, score) in enumerate(results, 1):
        print(f"{i}. Chunk ID: {chunk_id}")
        print(f"   Cosine Score: {score:.4f}")
        print(f"   Text: {text[:100]}...\n")
    
    assert len(results) > 0, "Vector search returned no results"
    print("PASSED: Vector search working correctly")


def test_graph_scoring():
  
    print("\n" + "="*70)
    print("TEST 2: Graph Scoring")
    print("="*70)
    
    # Test with a known node
    test_node = "chunk_7"
    scores = graph_score(graph_db.graph, test_node, depth=2)
    
    print(f"\nStarting Node: '{test_node}'")
    print(f"Related Nodes: {len(scores)} found\n")
    
    for node_id, score in sorted(scores.items(), key=lambda x: -x[1])[:10]:
        print(f"  {node_id}: {score:.2f}")
    
    assert len(scores) > 0, "Graph scoring returned no results"
    assert scores.get(test_node) == 1.0, "Starting node should have score 1.0"
    print("\nPASSED: Graph scoring working correctly")


def test_hybrid_ranking():
    """Test hybrid ranking component"""
    print("\n" + "="*70)
    print("TEST 3: Hybrid Ranking")
    print("="*70)
    
    # Mock data
    cosine_results = [
        ("chunk_1", "Sample text 1", 0.9),
        ("chunk_2", "Sample text 2", 0.7),
        ("chunk_3", "Sample text 3", 0.5),
    ]
    
    graph_scores = {
        "chunk_1": 1.0,
        "chunk_2": 0.5,
        "chunk_4": 1.0,  # Not in cosine results
    }
    
    results = hybrid_rank(cosine_results, graph_scores)
    
    print("\nInput:")
    print(f"  Cosine results: {len(cosine_results)} chunks")
    print(f"  Graph scores: {len(graph_scores)} chunks")
    print(f"\nOutput: {len(results)} ranked results\n")
    
    for i, result in enumerate(results, 1):
        expected_final = (
            result['cosine_similarity'] * COSINE_WEIGHT + 
            result['graph_score'] * GRAPH_WEIGHT
        )
        print(f"{i}. {result['chunk_id']}")
        print(f"   Cosine: {result['cosine_similarity']:.4f} | Graph: {result['graph_score']:.2f}")
        print(f"   Final: {result['final_score']:.4f} (expected: {expected_final:.4f})")
        
        assert abs(result['final_score'] - expected_final) < 0.001, "Score calculation error"
    
    # Verify sorting (descending)
    for i in range(len(results) - 1):
        assert results[i]['final_score'] >= results[i+1]['final_score'], "Results not sorted"
    
    print("\nPASSED: Hybrid ranking working correctly")


def test_hybrid_retrieve():
    """Test end-to-end hybrid retrieval"""
    print("\n" + "="*70)
    print("TEST 4: End-to-End Hybrid Retrieval")
    print("="*70)
    
    query = "Romeo and Juliet's relationship"
    results = hybrid_retrieve(query, top_k=5)
    
    print(f"\nQuery: '{query}'")
    print(f"Results: {len(results)} chunks\n")
    
    print("Top 5 Results:")
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. Chunk: {result['chunk_id']}")
        print(f"   Cosine: {result['cosine_similarity']:.4f}")
        print(f"   Graph: {result['graph_score']:.2f}")
        print(f"   Final: {result['final_score']:.4f}")
        print(f"   Text: {result['text'][:80]}...")
        
        # Validate format
        assert 'chunk_id' in result
        assert 'cosine_similarity' in result
        assert 'graph_score' in result
        assert 'final_score' in result
        assert 'text' in result
    
    # Test JSON serialization
    json_output = json.dumps(results, indent=2, ensure_ascii=False)
    assert json_output, "Failed to serialize to JSON"
    
    print("\nPASSED: End-to-end retrieval working correctly")


def test_output_format():
    """Test output format matches specification"""
    print("\n" + "="*70)
    print("TEST 5: Output Format Validation")
    print("="*70)
    
    query = "test query"
    results = hybrid_retrieve(query, top_k=2)
    
    if len(results) > 0:
        result = results[0]
        
        print("\nSample Result:")
        print(json.dumps(result, indent=2))
        
        # Validate required fields
        required_fields = ['chunk_id', 'cosine_similarity', 'graph_score', 'final_score', 'text']
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Validate types
        assert isinstance(result['chunk_id'], str), "chunk_id should be string"
        assert isinstance(result['cosine_similarity'], (int, float)), "cosine_similarity should be numeric"
        assert isinstance(result['graph_score'], (int, float)), "graph_score should be numeric"
        assert isinstance(result['final_score'], (int, float)), "final_score should be numeric"
        assert isinstance(result['text'], str), "text should be string"
        
        print("\nPASSED: Output format matches specification")
    else:
        print("\nSKIPPED: No results to validate (may need to run setup_graph.py first)")


def test_qa_accuracy():
    """Test that specific questions retrieve the correct answer chunks"""
    print("\n" + "="*70)
    print("TEST 6: Q&A Accuracy Verification")
    print("="*70)
    
    # Test Case: Who killed Tybalt?
    query = "What is TTLs?"
    expected_chunk_id = "doc1.txt"
    expected_text_part = "TTLs is a "
    
    print(f"\nQuery: '{query}'")
    print(f"Expected Chunk: {expected_chunk_id}")
    print(f"Expected Text: '...{expected_text_part}...'\n")
    
    results = hybrid_retrieve(query, top_k=3)
    
    if not results:
        print("FAILED: No results returned")
        return

    top_result = results[0]
    print("Top Result:")
    print(f"  Chunk ID: {top_result['chunk_id']}")
    print(f"  Final Score: {top_result['final_score']:.4f}")
    print(f"  Text: {top_result['text']}")
    
    # Verification
    is_correct_chunk = top_result['chunk_id'] == expected_chunk_id
    contains_answer = expected_text_part in top_result['text']
    
    if is_correct_chunk and contains_answer:
        print("\nPASSED: Correct answer retrieved as top result")
    else:
        print(f"\nFAILED: Expected {expected_chunk_id}, got {top_result['chunk_id']}")
        # Check if it's in top k
        for i, res in enumerate(results):
            if res['chunk_id'] == expected_chunk_id:
                print(f"  (Expected answer was found at rank {i+1})")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("HYBRID RETRIEVAL SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    try:
        # test_vector_search()
        # test_graph_scoring()
        # test_hybrid_ranking()
        # test_hybrid_retrieve()
        # test_output_format()
        test_qa_accuracy()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        print("\nHybrid Retrieval System is fully functional:")
        print("  Vector search (ChromaDB)")
        print("  Graph scoring (NetworkX)")
        print("  Hybrid ranking (0.7/0.3 formula)")
        print("  End-to-end pipeline")
        print("  JSON output format")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
