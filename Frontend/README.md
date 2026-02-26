# DevForge DB - Frontend UI

A modern, beautiful web interface for the Hybrid Vector + Graph Database System.

## 🎨 Features

- **Node Management**: Create, view, edit, and delete graph nodes
- **Edge Management**: Create and manage relationships between nodes
- **Advanced Search**: Vector search, graph traversal, and hybrid search with adjustable weights
- **PDF Processing**: Upload PDFs, automatically chunk and index them, then search within
- **Graph Visualization**: Interactive graph visualization using vis-network
- **Responsive Design**: Beautiful UI built with Tailwind CSS

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running on `http://localhost:8000`

### Installation

```bash
cd Frontend
npm install
```

### Development

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## 📁 Project Structure

```
Frontend/
├── src/
│   ├── components/      # Reusable UI components
│   │   └── Layout.tsx   # Main layout with sidebar
│   ├── pages/           # Page components
│   │   ├── Dashboard.tsx
│   │   ├── Nodes.tsx
│   │   ├── Edges.tsx
│   │   ├── Search.tsx
│   │   ├── PDFUpload.tsx
│   │   └── GraphView.tsx
│   ├── services/        # API service layer
│   │   └── api.ts       # Axios client and API functions
│   ├── App.tsx          # Main app component with routing
│   ├── main.tsx         # Entry point
│   └── index.css        # Tailwind CSS styles
├── package.json
├── vite.config.ts       # Vite configuration
└── tailwind.config.js   # Tailwind configuration
```

## 🔧 Configuration

### API URL

By default, the frontend connects to `http://localhost:8000`. To change this:

1. Create a `.env` file in the `Frontend/` directory
2. Add: `VITE_API_URL=http://your-api-url:8000`

### Proxy Configuration

The Vite dev server is configured to proxy `/api` requests to the backend. This is handled in `vite.config.ts`.

## 🎯 Usage

### Creating Nodes

1. Navigate to **Nodes** page
2. Click **Create Node**
3. Enter text content and metadata (JSON format)
4. Choose whether to auto-generate embeddings
5. Click **Create**

### Creating Edges

1. Navigate to **Edges** page
2. Click **Create Edge**
3. Enter source and target node IDs
4. Set relationship type and weight
5. Click **Create**

### Searching

1. Navigate to **Search** page
2. Choose search type (Vector or Hybrid)
3. Enter your query
4. Adjust parameters (top K, weights for hybrid search)
5. Click **Search**

### PDF Processing

1. Navigate to **PDF Upload** page
2. Drag and drop or browse for a PDF file
3. Enter a search query
4. Click **Upload & Search**
5. The PDF will be processed, chunked, indexed, and searched

### Graph Visualization

1. Navigate to **Graph View** page
2. View the interactive graph visualization
3. Drag nodes to rearrange
4. Click nodes to view details
5. Zoom and pan to explore

### Running Test Cases

1. Navigate to **Test Runner** page
2. Click **Run All Tests** to execute all test cases
3. Or click **Run** on individual test cases
4. View test results with detailed responses
5. Tests match the test cases from `devforge_test_case.py`

## 🛠️ Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **React Router** - Routing
- **TanStack Query** - Data fetching and caching
- **Axios** - HTTP client
- **vis-network** - Graph visualization
- **Lucide React** - Icons

## 📝 API Integration

The frontend uses the FastAPI backend endpoints:

- `GET/POST/PUT/DELETE /nodes` - Node CRUD
- `GET/POST/PUT/DELETE /edges` - Edge CRUD
- `POST /search/vector` - Vector search
- `GET /search/graph` - Graph traversal
- `POST /search/hybrid` - Hybrid search
- `POST /pdf/search` - PDF upload and search

## 🧪 Testing

The UI includes a comprehensive test runner that executes all test cases from `devforge_test_case.py`:

1. Navigate to **Test Runner** page (`/tests`)
2. Click **Run All Tests** to execute all 12 test cases sequentially
3. Or run individual tests by clicking the **Run** button on each test
4. View detailed results including:
   - Test status (passed/failed)
   - Response data
   - Error messages (if any)
   - Request payloads

### Test Cases Included

1. Create Node (doc1)
2. Get Node (doc1)
3. Update Node (doc1)
4. Create Node (doc4) - for edge testing
5. Create Edge (doc1 → doc4)
6. Get Edge
7. Update Edge (weight)
8. Vector Search with metadata filter
9. Graph Traversal from doc1
10. Hybrid Search
11. Delete Edge
12. Delete Node (doc7)

All test cases use the exact payloads from `devforge_test_case.py`.

See `src/services/api.ts` for the complete API client implementation.

## 🎨 Customization

### Colors

Edit `tailwind.config.js` to customize the color scheme. The primary color is currently set to blue.

### Styling

All styles use Tailwind CSS utility classes. Custom styles can be added to `src/index.css`.

## 🐛 Troubleshooting

### CORS Errors

Make sure the backend has CORS enabled (already configured in `app/main.py`).

### API Connection Issues

1. Verify the backend is running on `http://localhost:8000`
2. Check the browser console for errors
3. Verify the API URL in `.env` if using a custom URL

### Build Issues

1. Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
2. Clear Vite cache: `rm -rf .vite`

## 📚 Additional Notes

- The frontend uses React Query for efficient data fetching and caching
- All API calls are typed with TypeScript interfaces
- The UI is fully responsive and works on mobile devices
- Graph visualization requires graph data from the backend (add a GET /graph endpoint)

## 🚧 Future Enhancements

- [ ] Add authentication/authorization
- [ ] Real-time graph updates via WebSocket
- [ ] Export graph as image/PDF
- [ ] Advanced filtering and sorting
- [ ] Bulk operations for nodes/edges
- [ ] Search history
- [ ] Dark mode support

