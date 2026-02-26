# Graph Database Module

A lightweight, production-ready graph database implementation using NetworkX for hybrid Vector + Graph retrieval systems.

## 📋 Features

- **Node CRUD Operations**: Create, read, update, and delete graph nodes with text, metadata, and optional embeddings
- **Edge CRUD Operations**: Manage directed relationships between nodes with types and weights
- **Graph Traversal**: BFS traversal up to specified depth for related node discovery
- **Relevance Scoring**: Compute graph-based relevance scores using edge weights and hop distance
- **JSON Persistence**: Save and load entire graph structures
- **Production Ready**: Fully documented, type-hinted, and error-handled code

## 🏗️ Architecture

```
/graph_db/
    __init__.py         # Public API exports
    models.py           # GraphNode and GraphRelationship data models
    graph_db.py         # GraphDatabase core implementation
```

## 📦 Installation

The module requires Python 3.10+ and the following dependencies (already in `pyproject.toml`):

```toml
networkx = "^3.2.1"
```

Install dependencies using poetry:

```bash
poetry install --no-root
```

## 🚀 Quick Start

```python
from graph_db import GraphDatabase, GraphNode, GraphRelationship

# Initialize database
db = GraphDatabase()

# Create nodes
node1 = db.create_node(
    text="Python is a programming language",
    metadata={"source": "wikipedia", "category": "programming"}
)

node2 = db.create_node(
    text="Django is a Python web framework",
    metadata={"source": "docs", "category": "framework"}
)

# Create relationship
edge = db.create_edge(
    source_id=node2.id,
    target_id=node1.id,
    rel_type="uses",
    weight=2.0
)

# Traverse graph (BFS)
related_nodes = db.traverse(node1.id, depth=2)

# Compute relevance scores
scores = db.compute_graph_scores(node1.id, depth=2)

# Save to file
db.save("my_graph.json")

# Load from file
new_db = GraphDatabase()
new_db.load("my_graph.json")
```

## 📚 API Reference

### GraphNode

Represents a node in the graph.

**Attributes**:
- `id: str` - Unique identifier (UUID)
- `text: str` - Text content
- `metadata: dict` - Additional metadata
- `embedding: list[float] | None` - Optional vector embedding

**Methods**:
- `to_dict()` - Serialize to dictionary
- `from_dict(data)` - Deserialize from dictionary

### GraphRelationship

Represents an edge/relationship.

**Attributes**:
- `id: str` - Unique identifier (UUID)
- `source: str` - Source node ID
- `target: str` - Target node ID
- `type: str` - Relationship type
- `weight: float` - Edge weight for scoring

**Methods**:
- `to_dict()` - Serialize to dictionary
- `from_dict(data)` - Deserialize from dictionary

### GraphDatabase

Core database class using NetworkX DiGraph.

#### Persistence

- **`save(path: str)`** - Save graph to JSON file
- **`load(path: str)`** - Load graph from JSON file

#### Node CRUD

- **`create_node(text, metadata=None, embedding=None) -> GraphNode`** - Create new node
- **`get_node(node_id) -> GraphNode | None`** - Retrieve node by ID
- **`update_node(node_id, text=None, metadata=None, embedding=None) -> bool`** - Update node attributes
- **`delete_node(node_id) -> bool`** - Delete node and connected edges

#### Edge CRUD

- **`create_edge(source_id, target_id, rel_type, weight=1.0) -> GraphRelationship | None`** - Create relationship
- **`get_edge(edge_id) -> GraphRelationship | None`** - Retrieve edge by ID
- **`delete_edge(edge_id) -> bool`** - Delete relationship

#### Graph Operations

- **`traverse(start_id, depth) -> list[str]`** - BFS traversal returning reachable node IDs
- **`compute_graph_scores(start_id, depth) -> dict[str, float]`** - Calculate relevance scores

**Scoring Formula**: `score = (total_edge_weight) / (hop_distance)`

#### Utility

- **`get_stats() -> dict`** - Get node and edge counts

## 💡 Examples

See `examples.py` for comprehensive demonstrations including:

1. **Basic CRUD Operations** - Creating, reading, updating nodes and edges
2. **Graph Traversal** - BFS exploration at different depths
3. **Relevance Scoring** - Computing graph-based scores
4. **Persistence** - Saving and loading from JSON
5. **Knowledge Graph** - Real-world document retrieval scenario
6. **Delete Operations** - Removing nodes and edges

Run examples:

```bash
poetry run python examples.py
```

Or without poetry:

```bash
python examples.py
```

## 🔧 Testing

Syntax validation:

```bash
python -m py_compile graph_db/__init__.py
python -m py_compile graph_db/models.py
python -m py_compile graph_db/graph_db.py
python -m py_compile examples.py
```

## 🎯 Use Cases

### Hybrid Vector + Graph Retrieval

```python
# 1. Vector search finds top-k documents
vector_results = vector_db.search(query_embedding, k=10)

# 2. Expand using graph relationships
graph_db = GraphDatabase()
graph_db.load("knowledge_graph.json")

all_related = set()
for doc_id in vector_results:
    related = graph_db.traverse(doc_id, depth=2)
    all_related.update(related)

# 3. Re-rank using graph scores
graph_scores = {}
for doc_id in vector_results:
    scores = graph_db.compute_graph_scores(doc_id, depth=2)
    graph_scores.update(scores)

# 4. Combine vector and graph scores
final_scores = {}
for doc_id in all_related:
    vector_score = vector_scores.get(doc_id, 0)
    graph_score = graph_scores.get(doc_id, 0)
    final_scores[doc_id] = 0.7 * vector_score + 0.3 * graph_score
```

### Document Knowledge Graph

Build semantic relationships between documents:

```python
db = GraphDatabase()

# Add documents as nodes
doc1 = db.create_node("Introduction to ML", {"page": 1})
doc2 = db.create_node("Neural Networks", {"page": 10})

# Create semantic relationships
db.create_edge(doc2.id, doc1.id, "references", weight=2.5)
db.create_edge(doc2.id, doc1.id, "builds_on", weight=3.0)

# Save for persistence
db.save("knowledge_graph.json")
```

## 🔗 Integration with FastAPI

```python
from fastapi import FastAPI
from graph_db import GraphDatabase

app = FastAPI()
db = GraphDatabase()
db.load("knowledge_graph.json")

@app.get("/traverse/{node_id}")
async def traverse_graph(node_id: str, depth: int = 2):
    return {"nodes": db.traverse(node_id, depth)}

@app.get("/score/{node_id}")
async def get_scores(node_id: str, depth: int = 2):
    return db.compute_graph_scores(node_id, depth)
```

## 📊 Performance Characteristics

- **Node/Edge Insertion**: O(1)
- **Node/Edge Retrieval**: O(1)
- **BFS Traversal**: O(V + E) where V = nodes, E = edges within depth
- **Score Computation**: O(V + E) within depth
- **Persistence**: O(V + E) for save/load operations

## 🔒 Production Considerations

- All node and edge IDs are UUIDs for uniqueness
- MultiDiGraph support allows multiple edges between same nodes
- Bidirectional traversal (follows both incoming and outgoing edges)
- Error handling for missing nodes/edges
- Type hints and docstrings for all public methods
- JSON serialization for cross-platform compatibility

## 📝 License

Part of the Devforge DB project for hybrid vector + graph retrieval.

## 🤝 Contributing

For development:

```bash
# Activate poetry shell
poetry shell

# Install dependencies
poetry install --no-root

# Run tests
python test_import.py
python examples.py
```

## 📖 Further Reading

- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [Graph Databases for Retrieval](https://en.wikipedia.org/wiki/Graph_database)
- [Hybrid Search Systems](https://www.pinecone.io/learn/hybrid-search-intro/)
