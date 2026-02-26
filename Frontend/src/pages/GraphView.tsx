import { useEffect, useRef, useState } from 'react'
// @ts-ignore - vis-network includes types but they may not be properly exported
import { Network, DataSet } from 'vis-network/standalone'
import { RefreshCw, Loader } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import api from '../services/api'

interface GraphData {
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

export default function GraphView() {
  const containerRef = useRef<HTMLDivElement>(null)
  const networkRef = useRef<any>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  // Fetch graph data from backend
  const { data: graphData, isLoading, refetch } = useQuery<GraphData>({
    queryKey: ['graph-data'],
    queryFn: async () => {
      const response = await api.get('/graph')
      return response.data
    },
    refetchOnWindowFocus: false,
  })

  useEffect(() => {
    if (!containerRef.current || !graphData) return

    // Prepare nodes for vis-network
    const nodes = new DataSet<any>(
      graphData.nodes.map((node) => ({
        id: node.id,
        label: node.label || node.id,
        title: node.text,
        color: {
          border: '#2B7CE9',
          background: '#97C2FC',
          highlight: {
            border: '#2B7CE9',
            background: '#D2E5FF',
          },
        },
        font: {
          size: 14,
        },
        shape: 'dot',
        size: 20,
      }))
    )

    // Prepare edges for vis-network
    const edges = new DataSet<any>(
      graphData.edges.map((edge) => ({
        id: edge.id,
        from: edge.from,
        to: edge.to,
        label: edge.label || edge.type || '',
        title: `Weight: ${edge.weight}`,
        width: Math.max(1, edge.weight * 2),
        color: {
          color: '#848484',
          highlight: '#848484',
        },
        smooth: {
          type: 'continuous',
        },
      }))
    )

    const data = { nodes, edges }
    const options = {
      nodes: {
        shape: 'dot',
        size: 20,
        font: {
          size: 14,
        },
        borderWidth: 2,
        color: {
          border: '#2B7CE9',
          background: '#97C2FC',
          highlight: {
            border: '#2B7CE9',
            background: '#D2E5FF',
          },
        },
      },
      edges: {
        width: 2,
        color: {
          color: '#848484',
          highlight: '#848484',
        },
        smooth: {
          type: 'continuous',
        },
        arrows: {
          to: {
            enabled: true,
            scaleFactor: 0.5,
          },
        },
      },
      physics: {
        enabled: true,
        stabilization: {
          iterations: 200,
        },
        barnesHut: {
          gravitationalConstant: -2000,
          centralGravity: 0.1,
          springLength: 95,
          springConstant: 0.04,
          damping: 0.09,
        },
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true,
      },
    }

    // Destroy existing network if it exists
    if (networkRef.current) {
      networkRef.current.destroy()
    }

    const network = new Network(containerRef.current, data, options)
    networkRef.current = network

    // Handle node click
    network.on('click', (params: any) => {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0]
        setSelectedNode(nodeId)
      } else {
        setSelectedNode(null)
      }
    })

    // Handle double click to focus on node
    network.on('doubleClick', (params: any) => {
      if (params.nodes.length > 0) {
        network.focus(params.nodes[0], {
          scale: 1.5,
          animation: true,
        })
      }
    })

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy()
        networkRef.current = null
      }
    }
  }, [graphData])

  const handleRefresh = () => {
    refetch()
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Graph Visualization</h1>
          <p className="text-gray-600 mt-2">Interactive visualization of the knowledge graph</p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={isLoading}
          className="btn btn-secondary flex items-center space-x-2"
        >
          {isLoading ? (
            <Loader className="h-5 w-5 animate-spin" />
          ) : (
            <RefreshCw className="h-5 w-5" />
          )}
          <span>Refresh</span>
        </button>
      </div>

      {/* Graph Stats */}
      {graphData && (
        <div className="card">
          <div className="flex items-center space-x-6">
            <div>
              <p className="text-sm text-gray-600">Nodes</p>
              <p className="text-2xl font-bold text-gray-900">{graphData.stats.node_count}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Edges</p>
              <p className="text-2xl font-bold text-gray-900">{graphData.stats.edge_count}</p>
            </div>
          </div>
        </div>
      )}

      {/* Graph Visualization */}
      <div className="card p-0 overflow-hidden relative">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
            <div className="text-center">
              <Loader className="h-12 w-12 text-primary-600 animate-spin mx-auto" />
              <p className="mt-4 text-gray-600">Loading graph...</p>
            </div>
          </div>
        )}
        {!graphData && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50 z-10">
            <div className="text-center">
              <p className="text-gray-500">No graph data available</p>
              <p className="text-sm text-gray-400 mt-2">Create some nodes and edges to see the graph</p>
            </div>
          </div>
        )}
        <div
          ref={containerRef}
          className="w-full h-[600px] bg-gray-50"
          style={{ minHeight: '600px' }}
        />
      </div>

      {/* Selected Node Details */}
      {selectedNode && graphData && (
        <div className="card">
          <div className="flex justify-between items-start mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Node Details</h2>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>
          {(() => {
            const node = graphData.nodes.find((n) => n.id === selectedNode)
            if (!node) return null
            return (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Node ID</label>
                  <p className="text-gray-900 font-mono text-sm">{node.id}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Text Content</label>
                  <p className="text-gray-900 bg-gray-50 p-3 rounded-lg">{node.text}</p>
                </div>
                {node.metadata && Object.keys(node.metadata).length > 0 && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Metadata</label>
                    <pre className="text-sm bg-gray-50 p-3 rounded-lg overflow-auto">
                      {JSON.stringify(node.metadata, null, 2)}
                    </pre>
                  </div>
                )}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Connected Edges</label>
                  <div className="space-y-2">
                    {graphData.edges
                      .filter((e) => e.from === selectedNode || e.to === selectedNode)
                      .map((edge) => (
                        <div key={edge.id} className="bg-gray-50 p-3 rounded-lg">
                          <div className="flex items-center justify-between">
                            <div>
                              <span className="text-sm font-medium text-gray-900">
                                {edge.from === selectedNode ? '→' : '←'} {edge.from === selectedNode ? edge.to : edge.from}
                              </span>
                              <span className="ml-2 text-xs text-gray-500">({edge.type})</span>
                              <span className="ml-2 text-xs text-gray-500">weight: {edge.weight})</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    {graphData.edges.filter((e) => e.from === selectedNode || e.to === selectedNode).length === 0 && (
                      <p className="text-sm text-gray-500">No edges connected to this node</p>
                    )}
                  </div>
                </div>
              </div>
            )
          })()}
        </div>
      )}

      {/* Instructions */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">Instructions</h2>
        <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
          <li>Drag nodes to rearrange the graph</li>
          <li>Click on nodes to view details</li>
          <li>Double-click nodes to zoom in</li>
          <li>Zoom with mouse wheel or pinch gesture</li>
          <li>Pan by dragging the background</li>
          <li>Edge labels show relationship types</li>
          <li>Edge thickness represents weight</li>
        </ul>
      </div>
    </div>
  )
}
