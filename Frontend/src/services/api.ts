import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// ==================== Node Types ====================

export interface NodeCreate {
  id: string
  text: string
  metadata?: Record<string, any>
  embedding?: number[]
  regen_embedding?: boolean
}

export interface NodeCreateResponse {
  status: string
  id: string
  embedding_dim?: number
}

export interface Node {
  id: string
  text: string
  metadata: Record<string, any>
  embedding?: number[] | null
  edges?: Array<{
    edge_id: string
    target: string
    type: string
    weight: number
  }>
}

export interface NodeUpdate {
  text?: string
  metadata?: Record<string, any>
  regen_embedding?: boolean
}

export interface NodeUpdateResponse {
  status: string
  id: string
  embedding_regenerated: boolean
}

export interface NodeDeleteResponse {
  status: string
  id: string
  removed_edges_count: number
}

// ==================== Edge Types ====================

export interface EdgeCreate {
  source: string
  target: string
  type: string
  weight?: number
}

export interface EdgeCreateResponse {
  status: string
  edge_id: string
  source: string
  target: string
}

export interface Edge {
  edge_id: string
  source: string
  target: string
  type: string
  weight: number
}

export interface EdgeUpdate {
  weight: number
}

export interface EdgeUpdateResponse {
  status: string
  edge_id: string
  new_weight: number
}

export interface EdgeDeleteResponse {
  status: string
  edge_id: string
}

// ==================== Search Types ====================

export interface VectorSearchRequest {
  query_text: string
  top_k?: number
  metadata_filter?: Record<string, any>
}

export interface VectorSearchResultItem {
  id: string
  vector_score: number
}

export interface VectorSearchResponse {
  query_text: string
  results: VectorSearchResultItem[]
}

export interface GraphTraversalResponse {
  start_id: string
  depth: number
  nodes: Array<{
    id: string
    hop: number
    edge?: string
    weight?: number
    edge_path?: string[]
    weights?: number[]
  }>
}

export interface HybridSearchRequest {
  query_text: string
  vector_weight?: number
  graph_weight?: number
  top_k?: number
}

export interface HybridSearchResultItem {
  id: string
  vector_score: number
  graph_score: number
  final_score: number
  info: Record<string, any>
}

export interface HybridSearchResponse {
  query_text: string
  vector_weight: number
  graph_weight: number
  results: HybridSearchResultItem[]
}

export interface HybridSearchResult {
  node_id: string
  final_score: number
  cosine_similarity: number
  graph_score: number
  text: string
}

// ==================== API Functions ====================

// Node APIs
export const nodeApi = {
  create: (data: NodeCreate) => api.post<NodeCreateResponse>('/nodes', data),
  get: (id: string) => api.get<Node>(`/nodes/${id}`),
  update: (id: string, data: NodeUpdate) => api.put<NodeUpdateResponse>(`/nodes/${id}`, data),
  delete: (id: string) => api.delete<NodeDeleteResponse>(`/nodes/${id}`),
}

// Edge APIs
export const edgeApi = {
  create: (data: EdgeCreate) => api.post<EdgeCreateResponse>('/edges', data),
  get: (id: string) => api.get<Edge>(`/edges/${id}`),
  update: (id: string, data: EdgeUpdate) => api.put<EdgeUpdateResponse>(`/edges/${id}`, data),
  delete: (id: string) => api.delete<EdgeDeleteResponse>(`/edges/${id}`),
}

// Search APIs
export const searchApi = {
  vector: (data: VectorSearchRequest) => api.post<VectorSearchResponse>('/search/vector', data),
  graph: (startId: string, depth: number = 2, typeFilter?: string) => {
    const params = new URLSearchParams({ start_id: startId, depth: depth.toString() })
    if (typeFilter) params.append('type_filter', typeFilter)
    return api.get<GraphTraversalResponse>(`/search/graph?${params.toString()}`)
  },
  hybrid: (data: HybridSearchRequest) => api.post<HybridSearchResponse>('/search/hybrid', data),
}

// PDF API
export const pdfApi = {
  upload: (file: File, query: string) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('query', query)
    return api.post<HybridSearchResult>('/pdf/search', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },
}

// Graph API
export interface GraphData {
  nodes: Array<{
    id: string
    label: string
    text: string
    metadata: Record<string, any>
  }>
  edges: Array<{
    id: string
    from: string
    to: string
    label: string
    weight: number
    type: string
  }>
  stats: {
    node_count: number
    edge_count: number
  }
}

export const graphApi = {
  get: () => api.get<GraphData>('/graph'),
}

// Stats API
export interface StatsData {
  nodes: number
  edges: number
  vector_documents: number
  graph_stats: {
    nodes: number
    edges: number
  }
}

export const statsApi = {
  get: () => api.get<StatsData>('/stats'),
}

export default api
