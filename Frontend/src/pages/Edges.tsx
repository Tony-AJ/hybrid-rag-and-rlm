import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Trash2, Edit, Search } from 'lucide-react'
import { edgeApi, type Edge, type EdgeCreate, type EdgeUpdate } from '../services/api'

export default function Edges() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editingEdge, setEditingEdge] = useState<Edge | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const createMutation = useMutation({
    mutationFn: (data: EdgeCreate) => edgeApi.create(data).then(res => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edges'] })
      queryClient.invalidateQueries({ queryKey: ['graph-data'] }) // Refresh graph
      queryClient.invalidateQueries({ queryKey: ['stats'] }) // Refresh dashboard stats
      setShowCreateModal(false)
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: EdgeUpdate }) =>
      edgeApi.update(id, data).then(res => res.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edges'] })
      setEditingEdge(null)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => edgeApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['edges'] })
      queryClient.invalidateQueries({ queryKey: ['graph-data'] }) // Refresh graph
      queryClient.invalidateQueries({ queryKey: ['stats'] }) // Refresh dashboard stats
      setSelectedEdgeId(null)
    },
  })

  const handleCreate = (data: EdgeCreate) => {
    createMutation.mutate(data)
  }

  const handleUpdate = (id: string, data: EdgeUpdate) => {
    updateMutation.mutate({ id, data })
  }

  const handleDelete = (id: string) => {
    if (confirm('Are you sure you want to delete this edge?')) {
      deleteMutation.mutate(id)
    }
  }

  // Fetch edge details when selected
  const { data: selectedEdge } = useQuery({
    queryKey: ['edge', selectedEdgeId],
    queryFn: () => edgeApi.get(selectedEdgeId!).then(res => res.data),
    enabled: !!selectedEdgeId,
  })

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Edges</h1>
          <p className="text-gray-600 mt-2">Manage relationships between nodes</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn btn-primary flex items-center space-x-2"
        >
          <Plus className="h-5 w-5" />
          <span>Create Edge</span>
        </button>
      </div>

      {/* Search Bar */}
      <div className="card">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search edges by ID (enter edge ID to view details)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && searchQuery.trim()) {
                setSelectedEdgeId(searchQuery.trim())
              }
            }}
            className="input pl-10"
          />
        </div>
      </div>

      {/* Selected Edge Details */}
      {selectedEdge && (
        <div className="card">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Edge Details</h2>
              <p className="text-sm text-gray-500 mt-1">ID: {selectedEdge.edge_id}</p>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => setEditingEdge(selectedEdge)}
                className="btn btn-secondary flex items-center space-x-1"
              >
                <Edit className="h-4 w-4" />
                <span>Edit</span>
              </button>
              <button
                onClick={() => handleDelete(selectedEdge.edge_id)}
                className="btn btn-danger flex items-center space-x-1"
              >
                <Trash2 className="h-4 w-4" />
                <span>Delete</span>
              </button>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Source Node</label>
                <p className="text-gray-900 bg-gray-50 p-3 rounded-lg font-mono text-sm">
                  {selectedEdge.source}
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Target Node</label>
                <p className="text-gray-900 bg-gray-50 p-3 rounded-lg font-mono text-sm">
                  {selectedEdge.target}
                </p>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
                <p className="text-gray-900 bg-gray-50 p-3 rounded-lg">{selectedEdge.type}</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Weight</label>
                <p className="text-gray-900 bg-gray-50 p-3 rounded-lg">{selectedEdge.weight}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <EdgeModal
          onClose={() => setShowCreateModal(false)}
          onSubmit={handleCreate}
          isLoading={createMutation.isPending}
        />
      )}

      {/* Edit Modal */}
      {editingEdge && (
        <EdgeEditModal
          edge={editingEdge}
          onClose={() => setEditingEdge(null)}
          onSubmit={(data) => handleUpdate(editingEdge.edge_id, data)}
          isLoading={updateMutation.isPending}
        />
      )}
    </div>
  )
}

interface EdgeModalProps {
  onClose: () => void
  onSubmit: (data: EdgeCreate) => void
  isLoading: boolean
}

function EdgeModal({ onClose, onSubmit, isLoading }: EdgeModalProps) {
  const [source, setSource] = useState('')
  const [target, setTarget] = useState('')
  const [type, setType] = useState('related_to')
  const [weight, setWeight] = useState(1.0)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({ source, target, type, weight })
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Create Edge</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Source Node ID <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="input"
              required
              placeholder="Enter source node ID"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Target Node ID <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              className="input"
              required
              placeholder="Enter target node ID"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Relationship Type <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={type}
              onChange={(e) => setType(e.target.value)}
              className="input"
              required
              placeholder="e.g., related_to, contains, references"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Weight: {weight.toFixed(2)}
            </label>
            <input
              type="range"
              value={weight}
              onChange={(e) => setWeight(parseFloat(e.target.value))}
              min="0"
              max="2"
              step="0.1"
              className="w-full"
            />
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
              {isLoading ? 'Creating...' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

interface EdgeEditModalProps {
  edge: Edge
  onClose: () => void
  onSubmit: (data: EdgeUpdate) => void
  isLoading: boolean
}

function EdgeEditModal({ edge, onClose, onSubmit, isLoading }: EdgeEditModalProps) {
  const [weight, setWeight] = useState(edge.weight)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({ weight })
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Edit Edge</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="bg-gray-50 p-4 rounded-lg space-y-2">
            <div>
              <span className="text-sm font-medium text-gray-700">Source: </span>
              <span className="text-sm text-gray-900 font-mono">{edge.source}</span>
            </div>
            <div>
              <span className="text-sm font-medium text-gray-700">Target: </span>
              <span className="text-sm text-gray-900 font-mono">{edge.target}</span>
            </div>
            <div>
              <span className="text-sm font-medium text-gray-700">Type: </span>
              <span className="text-sm text-gray-900">{edge.type}</span>
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Weight: {weight.toFixed(2)}
            </label>
            <input
              type="range"
              value={weight}
              onChange={(e) => setWeight(parseFloat(e.target.value))}
              min="0"
              max="2"
              step="0.1"
              className="w-full"
            />
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
              {isLoading ? 'Updating...' : 'Update'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
