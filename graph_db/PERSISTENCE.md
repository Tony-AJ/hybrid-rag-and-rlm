# Graph Database - Auto-Persistence Update

## 🎯 New Feature: Automatic Persistence to `db/` Directory

The Graph Database now automatically saves data to the `db/` folder, just like ChromaDB!

## 📦 Usage

### Default Behavior (Auto-Persist)

```python
from graph_db import GraphDatabase

# Creates/loads from db/graph_data.json automatically
db = GraphDatabase()

# Add data - automatically saved after each operation!
node = db.create_node("My document", {"page": 1})
# File db/graph_data.json is updated automatically

# On next run, data is automatically loaded
db2 = GraphDatabase()  # Loads existing data from db/graph_data.json
```

### Custom Database Path

```python
# Use a custom path
db = GraphDatabase(db_path="db/my_custom_graph.json")

# Or store in a different directory
db = GraphDatabase(db_path="data/graphs/main.json")
```

### Disable Auto-Persistence (for testing)

```python
# Disable auto-save/load
db = GraphDatabase(auto_persist=False)

# Manually save when needed
db.create_node("test", {})
db.save()  # Saves to db/graph_data.json

# Or save to specific path
db.save("backup/graph.json")
```

## 🔄 Auto-Persistence Behavior

### On Initialization
- If `auto_persist=True` (default) and file exists → automatically loads data
- If file doesn't exist → starts with empty graph

### After Every Mutation
When `auto_persist=True`, data is automatically saved after:
- `create_node()`
- `update_node()`  
- `delete_node()`
- `create_edge()`
- `delete_edge()`

### Manual Persistence
```python
db.persist()  # Save to default path
# or
db.save()  # Same as persist()
# or  
db.save("custom/path.json")  # Save to specific path
```

## 📁 File Structure

```
project/
├── db/
│   └── graph_data.json      ← Automatically created and updated
├── graph_db/
│   ├── __init__.py
│   ├── models.py
│   └── graph_db.py
└── your_app.py
```

## 🎨 Examples

### Example 1: Simple Usage

```python
from graph_db import GraphDatabase

# First run - creates new database
db = GraphDatabase()
db.create_node("Document 1", {"source": "pdf"})
db.create_node("Document 2", {"source": "web"})
# Data saved to db/graph_data.json

# Later/different script - automatically loads existing data
db = GraphDatabase()
stats = db.get_stats()
print(f"Loaded {stats['nodes']} nodes")  # Output: Loaded 2 nodes
```

### Example 2: Multiple Databases

```python
# Different graphs for different purposes
docs_db = GraphDatabase(db_path="db/documents.json")
users_db = GraphDatabase(db_path="db/users.json")
concepts_db = GraphDatabase(db_path="db/concepts.json")
```

### Example 3: Disable for Testing

```python
def test_graph_operations():
    # Don't persist during tests
    db = GraphDatabase(auto_persist=False)
    
    # Your test code
    node = db.create_node("test", {})
    assert node is not None
    
    # No files created
```

## ⚡ Performance

- **Auto-save overhead**: Minimal (JSON write on each mutation)
- **For high-frequency writes**: Disable auto-persist and batch save
  
```python
# High-performance batch import
db = GraphDatabase(auto_persist=False)

for doc in documents:  # 10,000 documents
    db.create_node(doc.text, doc.metadata)

# Single save at the end
db.persist()
```

## 🔒 Thread Safety

- Not thread-safe by default
- For concurrent access, use external locking or separate instances

## 💡 Best Practices

1. **Default for apps**: Use auto-persist for simplicity
2. **Batch imports**: Disable auto-persist, save once at end
3. **Testing**: Always disable auto-persist
4. **Backups**: Periodically copy `db/graph_data.json`

## 🆚 Comparison with ChromaDB

| Feature | GraphDatabase | ChromaDB |
|---------|---------------|----------|
| Default location | `db/graph_data.json` | `db/chroma.sqlite3` |
| Format | JSON | SQLite |
| Auto-persist | Yes (configurable) | Yes |
| Human-readable | Yes | No |
| Size | Larger (JSON) | Smaller (binary) |

## 🔧 Migration from Old Code

If you have existing code using explicit save/load:

```python
# Old code
db = GraphDatabase()
db.create_node("test", {})
db.save("my_graph.json")

# New code (equivalent)
db = GraphDatabase(db_path="my_graph.json")
db.create_node("test", {})
# Automatically saved!
```

## 📄 License

Part of the Devforge DB hybrid vector + graph retrieval system.
