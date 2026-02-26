import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Upload, FileText, Search } from 'lucide-react'
import { pdfApi, nodeApi, type HybridSearchResult } from '../services/api'

export default function PDFUpload() {
  const [file, setFile] = useState<File | null>(null)
  const [query, setQuery] = useState('')
  const [dragActive, setDragActive] = useState(false)

  const uploadMutation = useMutation({
    mutationFn: ({ file, query }: { file: File; query: string }) =>
      pdfApi.upload(file, query).then(res => res.data),
  })

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile)
      } else {
        alert('Please upload a PDF file')
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!file || !query.trim()) {
      alert('Please select a PDF file and enter a search query')
      return
    }
    uploadMutation.mutate({ file, query })
  }

  // Fetch node details for the result
  const { data: resultNode } = useQuery({
    queryKey: ['node', uploadMutation.data?.node_id],
    queryFn: () => nodeApi.get(uploadMutation.data!.node_id).then(res => res.data),
    enabled: !!uploadMutation.data?.node_id,
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">PDF Upload & Search</h1>
        <p className="text-gray-600 mt-2">Upload a PDF, process it, and search within it</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File Upload */}
        <div className="card">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            PDF File
          </label>
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            {file ? (
              <div className="space-y-2">
                <FileText className="h-8 w-8 text-primary-600 mx-auto" />
                <p className="text-sm font-medium text-gray-900">{file.name}</p>
                <p className="text-xs text-gray-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
                <button
                  type="button"
                  onClick={() => setFile(null)}
                  className="text-sm text-red-600 hover:text-red-700"
                >
                  Remove
                </button>
              </div>
            ) : (
              <div>
                <p className="text-sm text-gray-600 mb-2">
                  Drag and drop a PDF file here, or click to browse
                </p>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="btn btn-secondary inline-block cursor-pointer"
                >
                  Browse Files
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Search Query */}
        <div className="card">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Search Query
          </label>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What would you like to search for in this PDF?"
              className="input pl-10"
              required
            />
          </div>
          <p className="text-xs text-gray-500 mt-2">
            The PDF will be processed, chunked, and indexed. Then a hybrid search will be performed.
          </p>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!file || !query.trim() || uploadMutation.isPending}
          className="btn btn-primary w-full flex items-center justify-center space-x-2"
        >
          {uploadMutation.isPending ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Processing PDF...</span>
            </>
          ) : (
            <>
              <Upload className="h-5 w-5" />
              <span>Upload & Search</span>
            </>
          )}
        </button>
      </form>

      {/* Results */}
      {uploadMutation.data && (
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Search Result</h2>
          <SearchResultCard 
            result={uploadMutation.data} 
            nodeData={resultNode}
          />
        </div>
      )}

      {uploadMutation.isError && (
        <div className="card bg-red-50 border-red-200">
          <p className="text-red-700">
            Error: {uploadMutation.error instanceof Error ? uploadMutation.error.message : 'Failed to process PDF'}
          </p>
        </div>
      )}
    </div>
  )
}

function SearchResultCard({ result, nodeData }: { result: HybridSearchResult; nodeData?: any }) {
  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex justify-between items-start mb-2">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-xs font-mono text-gray-500">ID: {result.node_id}</span>
            <span className="px-2 py-1 bg-primary-100 text-primary-700 text-xs rounded">
              Hybrid Search Result
            </span>
          </div>
          {nodeData ? (
            <p className="text-gray-900">{nodeData.text}</p>
          ) : (
            <p className="text-gray-900">{result.text || 'Loading node details...'}</p>
          )}
        </div>
      </div>
      <div className="mt-3 flex items-center space-x-4 text-sm">
        <div className="text-gray-600">
          <span className="font-medium">Vector Score:</span>{' '}
          {result.cosine_similarity.toFixed(4)}
        </div>
        <div className="text-gray-600">
          <span className="font-medium">Graph Score:</span> {result.graph_score.toFixed(4)}
        </div>
        <div className="text-primary-600 font-semibold">
          <span className="font-medium">Final Score:</span> {result.final_score.toFixed(4)}
        </div>
      </div>
      {nodeData?.metadata && Object.keys(nodeData.metadata).length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-100">
          <details className="text-xs">
            <summary className="cursor-pointer text-gray-500 hover:text-gray-700">
              Metadata
            </summary>
            <pre className="mt-2 text-xs bg-gray-50 p-2 rounded overflow-auto">
              {JSON.stringify(nodeData.metadata, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  )
}
