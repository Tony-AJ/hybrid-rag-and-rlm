import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const endpoints = {
  nodes: `/nodes`,
  edges: `/edges`,
  search: {
    vector: `/search/vector`,
    graph: `/search/graph`,
    hybrid: `/search/hybrid`,
  },
  pdf: `/pdf/search`,
  graph: `/graph`,
  stats: `/stats`,
}

const api = axios.create({
  baseURL: API_BASE,
  // headers: {
  //   'Content-Type': 'application/json',
  // },
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
  max_depth?: number
  path_distance_penalty?: number
}

export type RetrievalSource = 'vector' | 'graph' | (string & {})

export interface RetrievalReason {
  rank: number | null
  vector_rank: number | null
  graph_rank: number | null
  hop: number
  edge: string | null
  retrieval_method: RetrievalSource[]
}

export interface RetrievalMetadata {
  pipeline_version: string
  timestamp: string
  max_depth: number
  path_distance_penalty: number
  normalization: string
  distance_conversion: string
}

export interface HybridSearchResultItem {
  node_id: string
  /** True cosine similarity (1 - L2²/2). null when the node was only reached via the graph. */
  raw_vector_score: number | null
  /** Min-max normalized within this query's vector hits. null for graph-only nodes. */
  normalized_vector_score: number | null
  graph_score: number
  vector_contribution: number
  graph_contribution: number
  penalty_contribution: number
  /** Exact: vector_contribution + graph_contribution - penalty_contribution. May be negative. */
  unclamped_final_score: number
  /** max(0, unclamped_final_score) — the value used for ranking. */
  final_score: number
  retrieval_sources: RetrievalSource[]
  reason: RetrievalReason
  metadata: Record<string, any>
  text: string
  info: Record<string, any>
}

export interface HybridSearchResponse {
  query_text: string
  vector_weight: number
  graph_weight: number
  pipeline_version: string
  retrieval_metadata: RetrievalMetadata | null
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
  create: (data: NodeCreate) => api.post<NodeCreateResponse>(endpoints.nodes, data),
  get: (id: string) => api.get<Node>(`${endpoints.nodes}/${id}`),
  update: (id: string, data: NodeUpdate) => api.put<NodeUpdateResponse>(`${endpoints.nodes}/${id}`, data),
  delete: (id: string) => api.delete<NodeDeleteResponse>(`${endpoints.nodes}/${id}`),
}

// Edge APIs
export const edgeApi = {
  create: (data: EdgeCreate) => api.post<EdgeCreateResponse>(endpoints.edges, data),
  get: (id: string) => api.get<Edge>(`${endpoints.edges}/${id}`),
  update: (id: string, data: EdgeUpdate) => api.put<EdgeUpdateResponse>(`${endpoints.edges}/${id}`, data),
  delete: (id: string) => api.delete<EdgeDeleteResponse>(`${endpoints.edges}/${id}`),
}

// Search APIs
export const searchApi = {
  vector: (data: VectorSearchRequest) => api.post<VectorSearchResponse>(endpoints.search.vector, data),
  graph: (startId: string, depth: number = 2, typeFilter?: string) => {
    const params = new URLSearchParams({ start_id: startId, depth: depth.toString() })
    if (typeFilter) params.append('type_filter', typeFilter)
    return api.get<GraphTraversalResponse>(`${endpoints.search.graph}?${params.toString()}`)
  },
  hybrid: (data: HybridSearchRequest) => api.post<HybridSearchResponse>(endpoints.search.hybrid, data),
}

// PDF API
export const pdfApi = {
  upload: (file: File, query: string) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('query', query)
    return api.post<HybridSearchResult>(endpoints.pdf, formData, {
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
  get: () => api.get<GraphData>(endpoints.graph),
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
  get: () => api.get<StatsData>(endpoints.stats),
}

export default api

// ==================== RLM (retrieve-and-reason loop) ====================

export interface RlmRetrievalStatus {
  available: boolean
  detail?: string
  variant?: string
  description?: string
  alpha?: number
  beta?: number
  gamma?: number
  seed_k?: number
  max_depth?: number
  production_service_modified?: boolean
  control_metrics?: Record<string, number>
  selected_metrics?: Record<string, number>
  corpus_fingerprint?: string
  dataset_sha256?: string
}

export interface RlmProviderStatus {
  name: string
  hosted: boolean
  ready: boolean
  key_present: boolean
  evaluative: boolean
  models: string[]
  model_count?: number
  detail?: string
}

export interface RlmCorpus {
  corpus_id: string
  description?: string
  has_dataset: boolean
  privacy_restricted: boolean
  vector_doc_count?: number
}

export interface RlmStatus {
  retrieval: RlmRetrievalStatus
  providers: RlmProviderStatus[]
  corpora: RlmCorpus[]
  loop: { version: string; prompt_template_sha256: Record<string, string> }
}

export interface RlmDatasetQuery {
  id: number
  query: string
  category: string
  hops: number
  relevant: string[]
  gold_answer: { kind: string; entity_type?: string; entities?: string[]; predicates?: Record<string, any> }
}

export interface RlmAskRequest {
  question: string
  provider: string
  model?: string
  corpus_id?: string
  top_k?: number
  max_steps?: number
  max_retrievals?: number
  query_id?: number
}

export interface RlmAskResponse {
  question: string
  answer: string
  provider: string
  model: string
  evaluative: boolean
  stop_reason: string
  retrieval_calls: number
  generation_calls: number
  input_tokens: number
  output_tokens: number
  retrieval_latency_ms: number
  generation_latency_ms: number
  total_latency_ms: number
  steps: Array<{ step: number; query: string; retrieved: string[]; new_evidence: string[]; decision?: string }>
  evidence: Array<{ node_id: string; text: string; hop: number; retrieved_by_vector: boolean }>
  retrieval: { variant: string; weights: Record<string, number>; seed_k: number; max_depth: number; top_k: number }
  answer_evaluation?: Record<string, any> | null
  gold_answer?: Record<string, any> | null
}

export interface RlmExperimentSummary {
  file: string
  status?: string
  status_derived?: boolean
  recorded_status?: string
  evaluative?: boolean
  complete?: boolean
  queries_evaluated?: number
  queries_in_dataset?: number
  quota_stop?: { provider: string; detail: string; completed: number; remaining: number } | null
  caveat?: string | null
  provider?: string
  model?: string
  corpus_id?: string
  retrieval_variant?: string
  recorded_at?: string
  aggregate?: {
    answer_level: Record<string, number | null>
    chunk_level: Record<string, number>
    cost: Record<string, number>
  }
  error?: string
}

export const rlmApi = {
  status: () => api.get<RlmStatus>('/rlm/status'),
  dataset: (corpusId = 'meridian-transit-v1') =>
    api.get<{ corpus_id: string; dataset_version: string; count: number; queries: RlmDatasetQuery[] }>(
      '/rlm/dataset', { params: { corpus_id: corpusId } }),
  ask: (body: RlmAskRequest) => api.post<RlmAskResponse>('/rlm/ask', body),
  experiments: () => api.get<{ count: number; experiments: RlmExperimentSummary[] }>('/rlm/experiments'),
  experimentDetail: (file: string) => api.get<any>(`/rlm/experiments/${file}`),
  ablation: (corpusId = 'meridian-transit-v1') =>
    api.get<any>('/rlm/ablation', { params: { corpus_id: corpusId } }),
}
