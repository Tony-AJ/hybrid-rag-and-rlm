"""
Setup script to populate Graph Database with sample data for hybrid retrieval.

Creates a graph structure with chunk IDs as nodes and relationships between them.
"""

import os
from graph_db.graph_db import GraphDatabase


def setup_graph_database():
    """
    Initialize graph database with sample Romeo & Juliet chunk relationships.
    
    Creates:
        - Nodes representing text chunks
        - Edges representing relationships (parent-child, topic-subtopic, references)
    """
    # Initialize database
    db_path = os.path.join(os.path.dirname(__file__), "db", "graph_data.json")
    graph_db = GraphDatabase(db_path=db_path, auto_persist=True)
    
    print("\n" + "="*60)
    print("GRAPH DATABASE SETUP")
    print("="*60 + "\n")
    
    # Create sample nodes (representing chunks from Romeo & Juliet)
    nodes_data = [
        {"id": "chunk_romeo_death", "text": "Romeo believes Juliet is dead and drinks poison"},
        {"id": "chunk_juliet_death", "text": "Juliet wakes to find Romeo dead and stabs herself"},
        {"id": "chunk_prologue", "text": "Two households both alike in dignity, in fair Verona"},
        {"id": "chunk_balcony", "text": "Romeo and Juliet profess their love on the balcony"},
        {"id": "chunk_marriage", "text": "Romeo and Juliet are secretly married by Friar Lawrence"},
        {"id": "chunk_tybalt", "text": "Tybalt kills Mercutio, Romeo kills Tybalt in revenge"},
        {"id": "chunk_banishment", "text": "Romeo is banished from Verona for killing Tybalt"},
        {"id": "chunk_potion", "text": "Juliet drinks a potion to appear dead"},
        {"id": "chunk_tomb", "text": "Romeo goes to Juliet's tomb in the Capulet monument"},
        {"id": "chunk_families", "text": "The Montague and Capulet families are feuding"},
        {"id": "chunk_reconciliation", "text": "The families reconcile after the deaths of Romeo and Juliet"},
    ]
    
    print("Creating nodes...")
    for node_data in nodes_data:
        node = graph_db.create_node(
            text=node_data["text"],
            metadata={"source": "Romeo and Juliet", "type": "chunk"},
            embedding=None
        )
        # Update node with custom ID for easier reference
        graph_db.graph.remove_node(node.id)
        graph_db.graph.add_node(
            node_data["id"],
            text=node_data["text"],
            metadata={"source": "Romeo and Juliet", "type": "chunk"},
            embedding=None
        )
        print(f"  Created: {node_data['id']}")
    
    print("\nCreating relationships...")
    
    # Define relationships (edges)
    relationships = [
        # Death sequence
        ("chunk_potion", "chunk_juliet_death", "leads_to", 1.0),
        ("chunk_banishment", "chunk_romeo_death", "leads_to", 1.0),
        ("chunk_romeo_death", "chunk_juliet_death", "causes", 1.0),
        ("chunk_juliet_death", "chunk_reconciliation", "leads_to", 1.0),
        
        # Story progression
        ("chunk_prologue", "chunk_families", "introduces", 1.0),
        ("chunk_balcony", "chunk_marriage", "leads_to", 1.0),
        ("chunk_marriage", "chunk_tybalt", "followed_by", 0.8),
        ("chunk_tybalt", "chunk_banishment", "results_in", 1.0),
        ("chunk_banishment", "chunk_potion", "motivates", 0.9),
        ("chunk_potion", "chunk_tomb", "leads_to", 1.0),
        ("chunk_tomb", "chunk_romeo_death", "setting_for", 1.0),
        
        # Thematic connections
        ("chunk_families", "chunk_reconciliation", "resolved_by", 0.7),
        ("chunk_balcony", "chunk_romeo_death", "tragic_arc", 0.6),
        ("chunk_balcony", "chunk_juliet_death", "tragic_arc", 0.6),
    ]
    
    for source, target, rel_type, weight in relationships:
        edge = graph_db.create_edge(source, target, rel_type, weight)
        if edge:
            print(f"  ✓ {source} --[{rel_type}]--> {target} (weight: {weight})")
    
    # Persist to disk
    graph_db.persist()
    
    stats = graph_db.get_stats()
    print(f"\n{'='*60}")
    print(f"Graph Database Setup Complete!")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Saved to: {db_path}")
    print(f"{'='*60}\n")
    
    return graph_db


if __name__ == "__main__":
    setup_graph_database()
