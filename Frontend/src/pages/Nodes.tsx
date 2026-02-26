import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Trash2, Edit, Search, ExternalLink } from 'lucide-react'
import { nodeApi, type Node, type NodeCreate, type NodeUpdate } from '../services/api'

export default function Nodes() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editingNode, setEditingNode] = useState<Node | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const createMutation = useMutation({
    mutationFn: (data: NodeCreate) => nodeApi.create(data).then(res => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['nodes'] })
      queryClient.invalidateQueries({ queryKey: ['graph-data'] }) // Refresh graph
      queryClient.invalidateQueries({ queryKey: ['stats'] }) // Refresh dashboard stats
      setShowCreateModal(false)
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: NodeUpdate }) =>
      nodeApi.update(id, data).then(res => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['nodes'] })
      setEditingNode(null)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => nodeApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['nodes'] })
      queryClient.invalidateQueries({ queryKey: ['graph-data'] }) // Refresh graph
      queryClient.invalidateQueries({ queryKey: ['stats'] }) // Refresh dashboard stats
    },
  })

  const handleCreate = (data: NodeCreate) => {
    createMutation.mutate(data)
  }

  const handleUpdate = (id: string, data: NodeUpdate) => {
    updateMutation.mutate({ id, data })
  }

  const handleDelete = (id: string) => {
    if (confirm('Are you sure you want to delete this node? This will also remove all connected edges.')) {
      deleteMutation.mutate(id)
    }
  }

  // Fetch node details when selected
  const { data: selectedNode } = useQuery({
    queryKey: ['node', selectedNodeId],
    queryFn: () => nodeApi.get(selectedNodeId!).then(res => res.data),
    enabled: !!selectedNodeId,
  })

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Nodes</h1>
          <p className="text-gray-600 mt-2">Manage graph nodes and their content</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn btn-primary flex items-center space-x-2"
        >
          <Plus className="h-5 w-5" />
          <span>Create Node</span>
        </button>
      </div>

      {/* Search Bar */}
      <div className="card">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search nodes by ID (enter node ID to view details)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && searchQuery.trim()) {
                setSelectedNodeId(searchQuery.trim())
              }
            }}
            className="input pl-10"
          />
        </div>
      </div>

      {/* Selected Node Details */}
      {selectedNode && (
        <div className="card">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Node Details</h2>
              <p className="text-sm text-gray-500 mt-1">ID: {selectedNode.id}</p>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setEditingNode(selectedNode)}
                className="btn btn-secondary flex items-center space-x-1"
              >
                <Edit className="h-4 w-4" />
                <span>Edit</span>
              </button>
              <button
                onClick={() => handleDelete(selectedNode.id)}
                className="btn btn-danger flex items-center space-x-1"
              >
                <Trash2 className="h-4 w-4" />
                <span>Delete</span>
              </button>
            </div>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Text Content</label>
              <p className="text-gray-900 bg-gray-50 p-3 rounded-lg">{selectedNode.text}</p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Metadata</label>
              <pre className="text-sm bg-gray-50 p-3 rounded-lg overflow-auto">
                {JSON.stringify(selectedNode.metadata, null, 2)}
              </pre>
            </div>

            {selectedNode.edges && selectedNode.edges.length > 0 && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Connected Edges ({selectedNode.edges.length})</label>
                <div className="space-y-2">
                  {selectedNode.edges.map((edge, idx) => (
                    <div key={idx} className="bg-gray-50 p-3 rounded-lg flex items-center justify-between">
                      <div>
                        <span className="text-sm font-medium text-gray-900">
                          → {edge.target.slice(0, 8)}...
                        </span>
                        <span className="ml-2 text-xs text-gray-500">({edge.type})</span>
                        <span className="ml-2 text-xs text-gray-500">weight: {edge.weight}</span>
                      </div>
                      <a
                        href={`/edges/${edge.edge_id}`}
                        className="text-primary-600 hover:text-primary-700 text-sm flex items-center space-x-1"
                      >
                        <span>View Edge</span>
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <NodeModal
          onClose={() => setShowCreateModal(false)}
          onSubmit={handleCreate}
          isLoading={createMutation.isPending}
        />
      )}

      {/* Edit Modal */}
      {editingNode && (
        <NodeModal
          node={editingNode}
          onClose={() => setEditingNode(null)}
          onSubmit={(data) => handleUpdate(editingNode.id, data)}
          isLoading={updateMutation.isPending}
        />
      )}
    </div>
  )
}

interface NodeModalProps {
  node?: Node
  onClose: () => void
  onSubmit: (data: NodeCreate | NodeUpdate) => void
  isLoading: boolean
}

function NodeModal({ node, onClose, onSubmit, isLoading }: NodeModalProps) {
  const [id, setId] = useState(node?.id || '')
  const [text, setText] = useState(node?.text || '')
  const [metadata, setMetadata] = useState(JSON.stringify(node?.metadata || {}, null, 2))
  const [regenEmbedding, setRegenEmbedding] = useState(true)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const metadataObj = JSON.parse(metadata)
      if (node) {
        // Update
        onSubmit({ text, metadata: metadataObj, regen_embedding: regenEmbedding })
      } else {
        // Create - requires ID
        if (!id.trim()) {
          alert('Node ID is required')
          return
        }
        onSubmit({ id: id.trim(), text, metadata: metadataObj, regen_embedding: regenEmbedding })
      }
    } catch (error) {
      alert('Invalid JSON in metadata field')
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          {node ? 'Edit Node' : 'Create Node'}
        </h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          {!node && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Node ID <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={id}
                onChange={(e) => setId(e.target.value)}
                className="input"
                required
                placeholder="Enter unique node ID"
              />
              <p className="text-xs text-gray-500 mt-1">Must be a unique identifier</p>
            </div>
          )}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Text Content <span className="text-red-500">*</span>
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="input min-h-[100px]"
              required
              placeholder="Enter node text content..."
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Metadata (JSON)
            </label>
            <textarea
              value={metadata}
              onChange={(e) => setMetadata(e.target.value)}
              className="input font-mono text-sm min-h-[150px]"
              placeholder='{"source": "doc1", "category": "example"}'
            />
          </div>
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="regenEmbedding"
              checked={regenEmbedding}
              onChange={(e) => setRegenEmbedding(e.target.checked)}
              className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
            />
            <label htmlFor="regenEmbedding" className="text-sm text-gray-700">
              {node ? 'Regenerate embedding' : 'Generate embedding automatically'}
            </label>
          </div>
          <div className="flex justify-end space-x-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="btn btn-secondary"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={isLoading}>
              {isLoading ? 'Saving...' : node ? 'Update' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
