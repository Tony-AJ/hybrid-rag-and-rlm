import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Search as SearchIcon } from 'lucide-react'
import { searchApi, nodeApi } from '../services/api'
import type { HybridSearchResultItem } from '../services/api'

type SearchType = 'vector' | 'graph' | 'hybrid'

export default function Search() {
  const [searchType, setSearchType] = useState<SearchType>('hybrid')
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [vectorWeight, setVectorWeight] = useState(0.6)
  const [graphWeight, setGraphWeight] = useState(0.4)
  const [metadataFilter, setMetadataFilter] = useState('')
  const [graphStartId, setGraphStartId] = useState('')
  const [graphDepth, setGraphDepth] = useState(2)
  const [graphTypeFilter, setGraphTypeFilter] = useState('')

  const vectorMutation = useMutation({
    mutationFn: (query: string) =>
      searchApi.vector({
        query_text: query,
        top_k: topK,
        metadata_filter: metadataFilter ? JSON.parse(metadataFilter) : undefined,
      }).then(res => res.data),
  })

  const hybridMutation = useMutation({
    mutationFn: (query: string) =>
      searchApi.hybrid({
        query_text: query,
        top_k: topK,
        vector_weight: vectorWeight,
        graph_weight: graphWeight,
      }).then(res => res.data),
  })

  const graphQuery = useQuery({
    queryKey: ['graph-traversal', graphStartId, graphDepth, graphTypeFilter],
    queryFn: () => searchApi.graph(graphStartId, graphDepth, graphTypeFilter || undefined).then(res => res.data),
    enabled: !!graphStartId && graphStartId.trim() !== '',
  })

  const handleSearch = () => {
    if (!query.trim() && searchType !== 'graph') return

    if (searchType === 'vector') {
      vectorMutation.mutate(query)
    } else if (searchType === 'hybrid') {
      hybridMutation.mutate(query)
    }
  }

  const handleGraphSearch = () => {
    if (!graphStartId.trim()) return
    graphQuery.refetch()
  }

  const vectorResults = vectorMutation.data
  const hybridResults = hybridMutation.data
  const graphResults = graphQuery.data

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Search</h1>
        <p className="text-gray-600 mt-2">Perform vector, graph, or hybrid search</p>
      </div>

      {/* Search Type Selector */}
      <div className="card">
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setSearchType('vector')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${searchType === 'vector'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
          >
            Vector Search
          </button>
          <button
            onClick={() => setSearchType('graph')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${searchType === 'graph'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
          >
            Graph Traversal
          </button>
          <button
            onClick={() => setSearchType('hybrid')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${searchType === 'hybrid'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
          >
            Hybrid Search
          </button>
        </div>

        {/* Vector/Hybrid Search Input */}
        {(searchType === 'vector' || searchType === 'hybrid') && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Search Query
              </label>
              <div className="relative">
                <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Enter your search query..."
                  className="input pl-10"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Top K Results
                </label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
                  min="1"
                  max="50"
                  className="input"
                />
              </div>
            </div>

            {searchType === 'vector' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Metadata Filter (JSON, optional)
                </label>
                <textarea
                  value={metadataFilter}
                  onChange={(e) => setMetadataFilter(e.target.value)}
                  placeholder='{"source": "doc1"}'
                  className="input font-mono text-sm min-h-[80px]"
                />
                <p className="text-xs text-gray-500 mt-1">Leave empty for no filter</p>
              </div>
            )}

            {searchType === 'hybrid' && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Vector Weight: {vectorWeight.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    value={vectorWeight}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value)
                      setVectorWeight(val)
                      setGraphWeight(1 - val)
                    }}
                    min="0"
                    max="1"
                    step="0.1"
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Graph Weight: {graphWeight.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    value={graphWeight}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value)
                      setGraphWeight(val)
                      setVectorWeight(1 - val)
                    }}
                    min="0"
                    max="1"
                    step="0.1"
                    className="w-full"
                  />
                </div>
              </div>
            )}

            <button
              onClick={handleSearch}
              disabled={(vectorMutation.isPending || hybridMutation.isPending) || !query.trim()}
              className="btn btn-primary w-full flex items-center justify-center space-x-2"
            >
              <SearchIcon className="h-5 w-5" />
              <span>{(vectorMutation.isPending || hybridMutation.isPending) ? 'Searching...' : 'Search'}</span>
            </button>
          </div>
        )}

        {/* Graph Traversal Input */}
        {searchType === 'graph' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Start Node ID <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={graphStartId}
                onChange={(e) => setGraphStartId(e.target.value)}
                placeholder="Enter starting node ID"
                className="input"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Depth
                </label>
                <input
                  type="number"
                  value={graphDepth}
                  onChange={(e) => setGraphDepth(parseInt(e.target.value) || 2)}
                  min="1"
                  max="10"
                  className="input"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Type Filter (optional)
                </label>
                <input
                  type="text"
                  value={graphTypeFilter}
                  onChange={(e) => setGraphTypeFilter(e.target.value)}
                  placeholder="Filter by edge type"
                  className="input"
                />
              </div>
            </div>

            <button
              onClick={handleGraphSearch}
              disabled={graphQuery.isFetching || !graphStartId.trim()}
              className="btn btn-primary w-full flex items-center justify-center space-x-2"
            >
              <SearchIcon className="h-5 w-5" />
              <span>{graphQuery.isFetching ? 'Traversing...' : 'Traverse Graph'}</span>
            </button>
          </div>
        )}
      </div>

      {/* Vector Search Results */}
      {searchType === 'vector' && vectorResults && (
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Vector Search Results ({vectorResults.results.length})
          </h2>
          <p className="text-sm text-gray-600 mb-4">Query: "{vectorResults.query_text}"</p>
          <div className="space-y-4">
            {vectorResults.results.map((result, index) => (
              <VectorResultCard key={index} result={result} />
            ))}
          </div>
        </div>
      )}

      {/* Hybrid Search Results */}
      {searchType === 'hybrid' && hybridResults && (
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Hybrid Search Results ({hybridResults.results.length})
          </h2>
          <div className="mb-4 space-y-1 text-sm">
            <p>Query: "{hybridResults.query_text}"</p>
            <p>Vector Weight: {hybridResults.vector_weight} | Graph Weight: {hybridResults.graph_weight}</p>
            {hybridResults.retrieval_metadata && (
              <p className="text-xs text-gray-500">
                Pipeline v{hybridResults.retrieval_metadata.pipeline_version} |
                {' '}max_depth {hybridResults.retrieval_metadata.max_depth} |
                {' '}penalty {hybridResults.retrieval_metadata.path_distance_penalty} |
                {' '}{new Date(hybridResults.retrieval_metadata.timestamp).toLocaleString()}
              </p>
            )}
          </div>
          <div className="space-y-4">
            {hybridResults.results.map((result, index) => (
              <HybridResultCard key={index} result={result} />
            ))}
          </div>
        </div>
      )}

      {/* Graph Traversal Results */}
      {searchType === 'graph' && graphResults && (
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Graph Traversal Results ({graphResults.nodes.length} nodes)
          </h2>
          <div className="mb-4 space-y-1 text-sm">
            <p>Start ID: {graphResults.start_id}</p>
            <p>Depth: {graphResults.depth}</p>
          </div>
          <div className="space-y-2">
            {graphResults.nodes.map((node, index) => (
              <GraphNodeCard key={index} node={node} />
            ))}
          </div>
        </div>
      )}

      {/* Empty States */}
      {searchType === 'vector' && vectorResults && vectorResults.results.length === 0 && (
        <div className="card text-center py-8">
          <p className="text-gray-500">No results found</p>
        </div>
      )}

      {searchType === 'hybrid' && hybridResults && hybridResults.results.length === 0 && (
        <div className="card text-center py-8">
          <p className="text-gray-500">No results found</p>
        </div>
      )}

      {searchType === 'graph' && graphResults && graphResults.nodes.length === 0 && (
        <div className="card text-center py-8">
          <p className="text-gray-500">No nodes found in traversal</p>
        </div>
      )}
    </div>
  )
}

// Result Card Components
function VectorResultCard({ result }: { result: { id: string; vector_score: number } }) {
  const { data: node } = useQuery({
    queryKey: ['node', result.id],
    queryFn: () => nodeApi.get(result.id).then(res => res.data),
    enabled: !!result.id,
  })

  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:border-primary-300 transition-colors">
      <div className="flex justify-between items-start mb-2">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-xs font-mono text-gray-500">ID: {result.id}</span>
            <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
              Vector
            </span>
          </div>
          {node ? (
            <p className="text-gray-900">{node.text}</p>
          ) : (
            <p className="text-gray-400 italic">Loading node details...</p>
          )}
        </div>
      </div>
      <div className="mt-3 flex items-center space-x-4 text-sm">
        <div className="text-gray-600">
          <span className="font-medium">Vector Score:</span>{' '}
          {result.vector_score.toFixed(4)}
        </div>
      </div>
      {node?.metadata && Object.keys(node.metadata).length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-100">
          <details className="text-xs">
            <summary className="cursor-pointer text-gray-500 hover:text-gray-700">
              Metadata
            </summary>
            <pre className="mt-2 text-xs bg-gray-50 p-2 rounded overflow-auto">
              {JSON.stringify(node.metadata, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  )
}

const SOURCE_BADGE: Record<string, string> = {
  vector: 'bg-blue-100 text-blue-700',
  graph: 'bg-green-100 text-green-700',
}

/** null means the node was never scored by that retriever — show "—", not 0.0000. */
function score(value: number | null): string {
  return value === null ? '—' : value.toFixed(4)
}

function HybridResultCard({ result }: { result: HybridSearchResultItem }) {
  const { reason } = result
  // Backend returns the text inline; only fall back to a node fetch when it's empty.
  const { data: node } = useQuery({
    queryKey: ['node', result.node_id],
    queryFn: () => nodeApi.get(result.node_id).then(res => res.data),
    enabled: !!result.node_id && !result.text,
  })
  const text = result.text || node?.text
  const metadata = Object.keys(result.metadata).length > 0 ? result.metadata : node?.metadata

  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:border-primary-300 transition-colors">
      <div className="flex justify-between items-start mb-2">
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            {reason.rank !== null && (
              <span className="text-xs font-semibold text-gray-700">#{reason.rank}</span>
            )}
            <span className="text-xs font-mono text-gray-500">ID: {result.node_id}</span>
            {result.retrieval_sources.map(source => (
              <span
                key={source}
                className={`px-2 py-1 text-xs rounded ${SOURCE_BADGE[source] ?? 'bg-gray-100 text-gray-700'}`}
              >
                {source}
              </span>
            ))}
            <span className="text-xs text-gray-500">Hop: {reason.hop}</span>
            {reason.edge && <span className="text-xs text-gray-500">via {reason.edge}</span>}
          </div>
          {text ? (
            <p className="text-gray-900">{text}</p>
          ) : (
            <p className="text-gray-400 italic">Loading node details...</p>
          )}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <div className="text-gray-600">
          <span className="font-medium">Cosine:</span> {score(result.raw_vector_score)}
        </div>
        <div className="text-gray-600">
          <span className="font-medium">Normalized:</span> {score(result.normalized_vector_score)}
        </div>
        <div className="text-gray-600">
          <span className="font-medium">Graph:</span> {result.graph_score.toFixed(4)}
        </div>
        <div className="text-primary-600 font-semibold">
          <span className="font-medium">Final:</span> {result.final_score.toFixed(4)}
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-500">
        <span>vector +{result.vector_contribution.toFixed(4)}</span>
        <span>graph +{result.graph_contribution.toFixed(4)}</span>
        <span>penalty -{result.penalty_contribution.toFixed(4)}</span>
        <span className="text-gray-400">= {result.unclamped_final_score.toFixed(4)}</span>
        {result.unclamped_final_score < 0 && (
          <span className="text-amber-600">clamped to 0</span>
        )}
        {reason.vector_rank !== null && <span>vector rank #{reason.vector_rank}</span>}
        {reason.graph_rank !== null && <span>graph rank #{reason.graph_rank}</span>}
      </div>

      {metadata && Object.keys(metadata).length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-100">
          <details className="text-xs">
            <summary className="cursor-pointer text-gray-500 hover:text-gray-700">
              Metadata
            </summary>
            <pre className="mt-2 text-xs bg-gray-50 p-2 rounded overflow-auto">
              {JSON.stringify(metadata, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  )
}

function GraphNodeCard({ node }: { node: { id: string; hop: number; edge?: string; weight?: number; edge_path?: string[]; weights?: number[] } }) {
  const { data: nodeData } = useQuery({
    queryKey: ['node', node.id],
    queryFn: () => nodeApi.get(node.id).then(res => res.data),
    enabled: !!node.id,
  })

  return (
    <div className="border border-gray-200 rounded-lg p-3 hover:border-primary-300 transition-colors">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-1">
            <span className="text-xs font-mono text-gray-500">ID: {node.id}</span>
            <span className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded">
              Hop {node.hop}
            </span>
            {node.edge && (
              <span className="text-xs text-gray-500">via {node.edge}</span>
            )}
          </div>
          {nodeData ? (
            <p className="text-sm text-gray-900">{nodeData.text.substring(0, 100)}...</p>
          ) : (
            <p className="text-sm text-gray-400 italic">Loading...</p>
          )}
        </div>
      </div>
      {node.edge_path && node.edge_path.length > 0 && (
        <div className="mt-2 text-xs text-gray-500">
          Path: {node.edge_path.join(' → ')}
        </div>
      )}
    </div>
  )
}
