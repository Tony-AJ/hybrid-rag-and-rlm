

from graph_db import GraphDatabase
import os

print("="*60)
print("Graph Database Auto-Persistence Demo")
print("="*60)

print("\n1. Creating database with default path (db/graph_data.json)...")
db = GraphDatabase()  # Uses db/graph_data.json by default
print(f"   Database path: {db.db_path}")
print(f"   Auto-persist: {db.auto_persist}")

print("\n2. Adding some nodes...")
node1 = db.create_node(
    "Python is a programming language",
    metadata={"category": "programming"}
)
print(f"  Created node: {node1.id[:8]}... (auto-saved!)")

node2 = db.create_node(
    "Django is a Python framework",
    metadata={"category": "framework"}
)
print(f"  Created node: {node2.id[:8]}... (auto-saved!)")

print("\n3. Adding relationship...")
edge = db.create_edge(node2.id, node1.id, "uses", weight=2.0)
print(f"  Created edge: {edge.id[:8]}... (auto-saved!)")

print("\n4. Checking if file exists in db/...")
if os.path.exists("db/graph_data.json"):
    file_size = os.path.getsize("db/graph_data.json")
    print(f"  File exists: db/graph_data.json ({file_size} bytes)")
else:
    print("   File not found!")

print("\n5. Creating NEW database instance (should auto-load)...")
db2 = GraphDatabase()  # Auto-loads from db/graph_data.json
stats = db2.get_stats()
print(f"   Loaded graph: {stats['nodes']} nodes, {stats['edges']} edges")

print("\n6. Verifying data...")
for node_id in db2.graph.nodes():
    node = db2.get_node(node_id)
    print(f"   → {node.text}")

print("\n7. Adding another node to the loaded database...")
node3 = db2.create_node(
    "Flask is a lightweight framework",
    metadata={"category": "framework"}
)
print(f"  Created node: {node3.id[:8]}... (auto-saved!)")

print("\n8. Final stats...")
final_stats = db2.get_stats()
print(f"   Total: {final_stats['nodes']} nodes, {final_stats['edges']} edges")

print("\n" + "="*60)
print("Demo Complete!")
print("="*60)
print("\nThe graph database automatically persists to db/graph_data.json")
print("just like ChromaDB does. Check the db/ folder!")
print()
