import { useQuery } from '@tanstack/react-query'
import { Database, Network, Search, FileText, Loader } from 'lucide-react'
import { statsApi, graphApi } from '../services/api'

export default function Dashboard() {
  // Fetch real-time stats from backend
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery({
    queryKey: ['stats'],
    queryFn: () => statsApi.get().then(res => res.data),
    refetchInterval: 5000, // Refetch every 5 seconds for real-time updates
  })

  // Also fetch graph data for more detailed stats
  const { data: graphData } = useQuery({
    queryKey: ['graph-data'],
    queryFn: () => graphApi.get().then(res => res.data),
    refetchInterval: 5000,
  })

  // Calculate stats from data
  const statsData = {
    nodes: stats?.nodes || graphData?.stats.node_count || 0,
    edges: stats?.edges || graphData?.stats.edge_count || 0,
    vectorDocuments: stats?.vector_documents || 0,
    searches: 0, // This would need a separate tracking mechanism
    pdfs: 0, // This would need a separate tracking mechanism
  }

  const statCards = [
    { 
      label: 'Total Nodes', 
      value: statsData.nodes, 
      icon: Database, 
      color: 'bg-blue-500',
      loading: statsLoading && !stats
    },
    { 
      label: 'Total Edges', 
      value: statsData.edges, 
      icon: Network, 
      color: 'bg-green-500',
      loading: statsLoading && !stats
    },
    { 
      label: 'Vector Documents', 
      value: statsData.vectorDocuments, 
      icon: Search, 
      color: 'bg-purple-500',
      loading: statsLoading && !stats
    },
    { 
      label: 'Graph Connections', 
      value: statsData.edges, 
      icon: FileText, 
      color: 'bg-orange-500',
      loading: statsLoading && !stats
    },
  ]

  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-2">Overview of your hybrid vector + graph database</p>
        </div>
        <button
          onClick={() => refetchStats()}
          disabled={statsLoading}
          className="btn btn-secondary flex items-center space-x-2"
        >
          {statsLoading ? (
            <Loader className="h-5 w-5 animate-spin" />
          ) : (
            <span>Refresh</span>
          )}
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat) => {
          const Icon = stat.icon
          return (
            <div key={stat.label} className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.label}</p>
                  {stat.loading ? (
                    <div className="flex items-center space-x-2 mt-2">
                      <Loader className="h-6 w-6 text-gray-400 animate-spin" />
                      <span className="text-gray-400">Loading...</span>
                    </div>
                  ) : (
                    <p className="text-3xl font-bold text-gray-900 mt-2">{stat.value}</p>
                  )}
                </div>
                <div className={`${stat.color} p-3 rounded-lg`}>
                  <Icon className="h-6 w-6 text-white" />
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <a
            href="/nodes"
            className="p-4 border border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors"
          >
            <Database className="h-6 w-6 text-primary-600 mb-2" />
            <h3 className="font-medium text-gray-900">Create Node</h3>
            <p className="text-sm text-gray-600 mt-1">Add a new node to the graph</p>
          </a>
          <a
            href="/search"
            className="p-4 border border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors"
          >
            <Search className="h-6 w-6 text-primary-600 mb-2" />
            <h3 className="font-medium text-gray-900">Search</h3>
            <p className="text-sm text-gray-600 mt-1">Perform vector, graph, or hybrid search</p>
          </a>
          <a
            href="/pdf"
            className="p-4 border border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors"
          >
            <FileText className="h-6 w-6 text-primary-600 mb-2" />
            <h3 className="font-medium text-gray-900">Upload PDF</h3>
            <p className="text-sm text-gray-600 mt-1">Process and index a PDF document</p>
          </a>
        </div>
      </div>

      {/* System Info */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">System Architecture</h2>
        <div className="space-y-3 text-sm text-gray-600">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span>Vector Database: ChromaDB with HuggingFace embeddings</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span>Graph Database: NetworkX MultiDiGraph</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
            <span>Hybrid Retrieval: Weighted combination of vector + graph scores</span>
          </div>
        </div>
      </div>

      {/* Real-time Status */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">System Status</h2>
        <div className="space-y-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Data Refresh</span>
            <span className="text-green-600 font-medium">Auto (every 5s)</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Last Updated</span>
            <span className="text-gray-900">
              {stats ? new Date().toLocaleTimeString() : 'Never'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
