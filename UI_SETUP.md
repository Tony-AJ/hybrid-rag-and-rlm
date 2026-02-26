# 🎨 UI Setup Guide

A complete modern web UI has been created for your Hybrid Vector + Graph Database System!

## ✨ What's Included

- **Modern React + TypeScript** frontend with Vite
- **Beautiful Tailwind CSS** styling
- **Full CRUD operations** for nodes and edges
- **Advanced search** (Vector, Graph, Hybrid)
- **PDF upload and processing** interface
- **Interactive graph visualization**
- **Responsive design** that works on all devices

## 🚀 Quick Start

### 1. Install Frontend Dependencies

```bash
cd Frontend
npm install
```

### 2. Start the Backend API

Make sure your FastAPI backend is running:

```bash
# From the project root
poetry run uvicorn app.main:app --reload
```

The API should be available at `http://localhost:8000`

### 3. Start the Frontend

```bash
# From the Frontend directory
npm run dev
```

The UI will be available at `http://localhost:3000`

## 📋 Features

### Dashboard
- Overview of system statistics
- Quick action buttons
- System architecture information

### Nodes Management
- Create nodes with text and metadata
- View all nodes (when API endpoint is added)
- Edit existing nodes
- Delete nodes
- Search/filter nodes

### Edges Management
- Create relationships between nodes
- Set relationship types and weights
- View all edges (when API endpoint is added)

### Search
- **Vector Search**: Semantic similarity search
- **Hybrid Search**: Combined vector + graph search with adjustable weights
- Real-time results with detailed scoring
- Adjustable top-K results

### PDF Upload
- Drag-and-drop PDF upload
- Automatic text extraction and chunking
- Automatic indexing
- Search within uploaded PDFs

### Graph Visualization
- Interactive graph visualization using vis-network
- Drag nodes to rearrange
- Zoom and pan
- Click nodes for details

## 🔧 Configuration

### CORS Setup

CORS has already been configured in `app/main.py` to allow requests from `http://localhost:3000`.

### API URL

By default, the frontend connects to `http://localhost:8000`. To change this:

1. Create a `.env` file in the `Frontend/` directory
2. Add: `VITE_API_URL=http://your-api-url:8000`

## 📁 Project Structure

```
Frontend/
├── src/
│   ├── components/          # Reusable components
│   │   └── Layout.tsx       # Main layout with sidebar
│   ├── pages/               # Page components
│   │   ├── Dashboard.tsx
│   │   ├── Nodes.tsx
│   │   ├── Edges.tsx
│   │   ├── Search.tsx
│   │   ├── PDFUpload.tsx
│   │   └── GraphView.tsx
│   ├── services/            # API integration
│   │   └── api.ts           # Axios client
│   ├── App.tsx              # Main app
│   ├── main.tsx             # Entry point
│   └── index.css            # Styles
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

## 🎯 Usage Examples

### Creating a Node

1. Navigate to **Nodes** page
2. Click **Create Node**
3. Enter text: "Machine learning is a subset of AI"
4. Add metadata: `{"source": "doc1", "category": "AI"}`
5. Enable auto-embedding
6. Click **Create**

### Creating an Edge

1. Navigate to **Edges** page
2. Click **Create Edge**
3. Enter source and target node IDs
4. Set type: "related_to"
5. Adjust weight slider
6. Click **Create**

### Performing a Hybrid Search

1. Navigate to **Search** page
2. Select **Hybrid Search**
3. Enter query: "What is machine learning?"
4. Adjust vector/graph weights (default: 0.7/0.3)
5. Set top K results
6. Click **Search**

### Uploading and Searching a PDF

1. Navigate to **PDF Upload** page
2. Drag and drop a PDF file
3. Enter search query
4. Click **Upload & Search**
5. View results with hybrid scoring

## 🛠️ Tech Stack

- **React 18** - Modern UI framework
- **TypeScript** - Type safety
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first CSS
- **React Router** - Client-side routing
- **TanStack Query** - Data fetching
- **Axios** - HTTP client
- **vis-network** - Graph visualization
- **Lucide React** - Beautiful icons

## 🐛 Troubleshooting

### CORS Errors

If you see CORS errors:
1. Make sure the backend is running
2. Verify CORS is enabled in `app/main.py` (already done)
3. Check browser console for specific errors

### API Connection Issues

1. Verify backend is running on port 8000
2. Check `http://localhost:8000/docs` to see API docs
3. Verify API URL in `.env` file if using custom URL

### Build Issues

```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf .vite
```

## 📝 Next Steps

### Optional API Endpoints to Add

For full functionality, consider adding these endpoints:

1. **GET /nodes** - List all nodes
2. **GET /edges** - List all edges
3. **GET /graph** - Get graph data for visualization
4. **GET /stats** - Get system statistics

### Enhancements

- Add authentication/authorization
- Real-time updates via WebSocket
- Export graph as image
- Advanced filtering
- Dark mode
- Search history

## 🎉 You're All Set!

The UI is ready to use. Start both the backend and frontend, then navigate to `http://localhost:3000` to see your beautiful new interface!

For more details, see `Frontend/README.md`.

