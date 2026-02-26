# Graph Visualization & Backend Integration Update

## ✅ Changes Made

### 1. Backend - New Graph Endpoint

**Created:** `app/routers/graph.py`

- **Endpoint:** `GET /graph`
- **Purpose:** Returns all nodes and edges for graph visualization
- **Response Format:**
  ```json
  {
    "nodes": [
      {
        "id": "doc1",
        "label": "Redis caching strategies...",
        "text": "Redis caching strategies",
        "metadata": {...}
      }
    ],
    "edges": [
      {
        "id": "edge_123",
        "from": "doc1",
        "to": "doc4",
        "label": "related_to",
        "weight": 0.8,
        "type": "related_to"
      }
    ],
    "stats": {
      "node_count": 5,
      "edge_count": 3
    }
  }
  ```

**Updated:** `app/main.py`
- Added graph router to the FastAPI app

### 2. Frontend - Graph Visualization

**Updated:** `Frontend/src/pages/GraphView.tsx`

- ✅ Fetches real graph data from `/graph` endpoint
- ✅ Displays all nodes and edges from the backend
- ✅ Shows node statistics (count of nodes and edges)
- ✅ Interactive features:
  - Click nodes to view details
  - Double-click to zoom in
  - Drag nodes to rearrange
  - Edge labels show relationship types
  - Edge thickness represents weight
- ✅ Auto-refreshes when nodes/edges are created/deleted
- ✅ Loading states and empty states

**Updated:** `Frontend/src/services/api.ts`
- Added `GraphData` interface
- Added `graphApi.get()` function

### 3. Auto-Refresh Integration

**Updated:** `Frontend/src/pages/Nodes.tsx`
- Graph automatically refreshes when nodes are created/deleted

**Updated:** `Frontend/src/pages/Edges.tsx`
- Graph automatically refreshes when edges are created/deleted

## 📋 Request/Response Formats Verified

### Node Creation (Matches README)
**Request:**
```json
{
  "id": "doc1",
  "text": "Redis caching strategies",
  "metadata": { "type": "article", "tags": ["cache", "redis"] },
  "embedding": null,
  "regen_embedding": true
}
```

**Response:**
```json
{
  "status": "created",
  "id": "doc1",
  "embedding_dim": 384
}
```

### Edge Creation (Matches README)
**Request:**
```json
{
  "source": "doc1",
  "target": "doc4",
  "type": "related_to",
  "weight": 0.8
}
```

**Response:**
```json
{
  "status": "created",
  "edge_id": "edge_123",
  "source": "doc1",
  "target": "doc4"
}
```

## 🎯 How It Works

1. **Create Nodes/Edges:**
   - User creates nodes or edges through the UI
   - Forms match README specifications exactly
   - Data is sent to backend API

2. **Graph Visualization:**
   - Graph page fetches all nodes and edges from `/graph` endpoint
   - vis-network renders the graph interactively
   - Graph automatically refreshes when data changes

3. **Real-time Updates:**
   - When nodes/edges are created/deleted, the graph query is invalidated
   - Graph automatically refetches and updates the visualization

## 🚀 Usage

1. **Start Backend:**
   ```bash
   poetry run uvicorn app.main:app --reload
   ```

2. **Start Frontend:**
   ```bash
   cd Frontend
   npm run dev
   ```

3. **Create Nodes:**
   - Go to Nodes page
   - Click "Create Node"
   - Fill in ID, text, metadata (JSON)
   - Enable "Generate embedding automatically"
   - Click Create

4. **Create Edges:**
   - Go to Edges page
   - Click "Create Edge"
   - Enter source and target node IDs
   - Set type and weight
   - Click Create

5. **View Graph:**
   - Go to Graph View page
   - See all nodes and edges visualized
   - Click nodes to see details
   - Graph updates automatically when you create/delete nodes/edges

## ✨ Features

- ✅ Real-time graph visualization
- ✅ Automatic graph refresh on data changes
- ✅ Node and edge details on click
- ✅ Interactive graph manipulation (drag, zoom, pan)
- ✅ Edge labels and weights displayed
- ✅ Statistics display (node/edge counts)
- ✅ Matches README API specifications exactly

## 📝 Notes

- Graph endpoint returns all nodes and edges in the database
- Graph visualization uses vis-network library
- Edge thickness represents weight
- Node labels show truncated text (first 50 chars)
- Full node text available in details panel


