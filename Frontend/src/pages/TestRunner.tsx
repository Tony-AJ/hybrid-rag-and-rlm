import { useState } from 'react'
import { Play, CheckCircle, XCircle, Loader, RefreshCw } from 'lucide-react'
import { nodeApi, edgeApi, searchApi, type NodeCreate, type NodeUpdate, type EdgeCreate, type EdgeUpdate } from '../services/api'

interface TestCase {
  id: string
  name: string
  description: string
  endpoint: string
  method: string
  payload?: any
  expectedStatus?: number
}

interface TestResult {
  testId: string
  status: 'pending' | 'running' | 'passed' | 'failed'
  message?: string
  response?: any
  error?: string
}

const testCases: TestCase[] = [
  {
    id: '1',
    name: 'Create Node',
    description: 'Create a node with id "doc1"',
    endpoint: '/nodes',
    method: 'POST',
    payload: {
      id: 'doc1',
      text: 'Redis caching strategies',
      metadata: { type: 'article', tags: ['cache', 'redis'] },
      embedding: null,
      regen_embedding: true,
    },
    expectedStatus: 200,
  },
  {
    id: '2',
    name: 'Get Node',
    description: 'Get node with id "doc1"',
    endpoint: '/nodes/doc1',
    method: 'GET',
    expectedStatus: 200,
  },
  {
    id: '3',
    name: 'Update Node',
    description: 'Update node "doc1"',
    endpoint: '/nodes/doc1',
    method: 'PUT',
    payload: {
      text: 'Updated redis caching guide',
      metadata: { type: 'guide' },
      regen_embedding: true,
    },
    expectedStatus: 200,
  },
  {
    id: '4',
    name: 'Create Node (doc4)',
    description: 'Create a node with id "doc4" for edge testing',
    endpoint: '/nodes',
    method: 'POST',
    payload: {
      id: 'doc4',
      text: 'Database optimization techniques',
      metadata: { type: 'article', tags: ['database', 'optimization'] },
      embedding: null,
      regen_embedding: true,
    },
    expectedStatus: 200,
  },
  {
    id: '5',
    name: 'Create Edge',
    description: 'Create an edge between doc1 and doc4',
    endpoint: '/edges',
    method: 'POST',
    payload: {
      source: 'doc1',
      target: 'doc4',
      type: 'related_to',
      weight: 0.8,
    },
    expectedStatus: 200,
  },
  {
    id: '6',
    name: 'Get Edge',
    description: 'Get the created edge (will use edge_id from previous test)',
    endpoint: '/edges/{edge_id}',
    method: 'GET',
    expectedStatus: 200,
  },
  {
    id: '7',
    name: 'Update Edge',
    description: 'Update edge weight (will use edge_id from previous test)',
    endpoint: '/edges/{edge_id}',
    method: 'PUT',
    payload: {
      weight: 0.95,
    },
    expectedStatus: 200,
  },
  {
    id: '8',
    name: 'Vector Search',
    description: 'Search for "redis caching" with metadata filter',
    endpoint: '/search/vector',
    method: 'POST',
    payload: {
      query_text: 'redis caching',
      top_k: 5,
      metadata_filter: { type: 'guide' },
    },
    expectedStatus: 200,
  },
  {
    id: '9',
    name: 'Graph Traversal',
    description: 'Traverse graph from doc1 with depth 2',
    endpoint: '/search/graph?start_id=doc1&depth=2',
    method: 'GET',
    expectedStatus: 200,
  },
  {
    id: '10',
    name: 'Hybrid Search',
    description: 'Perform hybrid search for "redis caching"',
    endpoint: '/search/hybrid',
    method: 'POST',
    payload: {
      query_text: 'redis caching',
      vector_weight: 0.6,
      graph_weight: 0.4,
      top_k: 5,
    },
    expectedStatus: 200,
  },
  {
    id: '11',
    name: 'Delete Edge',
    description: 'Delete the created edge (will use edge_id from previous test)',
    endpoint: '/edges/{edge_id}',
    method: 'DELETE',
    expectedStatus: 200,
  },
  {
    id: '12',
    name: 'Delete Node',
    description: 'Delete node doc7 (if exists)',
    endpoint: '/nodes/doc7',
    method: 'DELETE',
    expectedStatus: 200,
  },
]

export default function TestRunner() {
  const [results, setResults] = useState<Record<string, TestResult>>({})
  const [isRunning, setIsRunning] = useState(false)
  const [currentTestId, setCurrentTestId] = useState<string | null>(null)
  const [edgeId, setEdgeId] = useState<string | null>(null)
  const [edgeIdDisplay, setEdgeIdDisplay] = useState<string | null>(null)

  const updateResult = (testId: string, result: Partial<TestResult>) => {
    setResults((prev) => ({
      ...prev,
      [testId]: { ...prev[testId], ...result } as TestResult,
    }))
  }

  const runTest = async (test: TestCase): Promise<TestResult> => {
    updateResult(test.id, { status: 'running' })

    try {
      let response: any

      // Handle dynamic endpoints
      let endpoint = test.endpoint
      // Note: edge_id replacement is handled in the switch cases below

      switch (test.method) {
        case 'POST':
          if (test.endpoint === '/nodes') {
            response = await nodeApi.create(test.payload as NodeCreate)
          } else if (test.endpoint === '/edges') {
            response = await edgeApi.create(test.payload as EdgeCreate)
            // Store edge_id for subsequent tests
            if (response.data.edge_id) {
              setEdgeId(response.data.edge_id)
              setEdgeIdDisplay(response.data.edge_id)
            }
          } else if (test.endpoint === '/search/vector') {
            response = await searchApi.vector(test.payload)
          } else if (test.endpoint === '/search/hybrid') {
            response = await searchApi.hybrid(test.payload)
          }
          break

        case 'GET':
          if (endpoint.startsWith('/nodes/')) {
            const nodeId = endpoint.split('/nodes/')[1]
            response = await nodeApi.get(nodeId)
          } else if (endpoint.startsWith('/edges/')) {
            const edgeIdParam = endpoint.split('/edges/')[1]
            // Use stored edge_id if endpoint has placeholder
            const edgeIdToUse = edgeIdParam === '{edge_id}' && edgeId ? edgeId : edgeIdParam
            if (!edgeIdToUse || edgeIdToUse === '{edge_id}') {
              throw new Error('Edge ID not available. Run "Create Edge" test first.')
            }
            response = await edgeApi.get(edgeIdToUse)
          } else if (endpoint.startsWith('/search/graph')) {
            const params = new URLSearchParams(endpoint.split('?')[1])
            const startId = params.get('start_id') || ''
            const depth = parseInt(params.get('depth') || '2')
            response = await searchApi.graph(startId, depth)
          }
          break

        case 'PUT':
          if (endpoint.startsWith('/nodes/')) {
            const nodeId = endpoint.split('/nodes/')[1]
            response = await nodeApi.update(nodeId, test.payload as NodeUpdate)
          } else if (endpoint.startsWith('/edges/')) {
            const edgeIdParam = endpoint.split('/edges/')[1]
            // Use stored edge_id if endpoint has placeholder
            const edgeIdToUse = edgeIdParam === '{edge_id}' && edgeId ? edgeId : edgeIdParam
            if (!edgeIdToUse || edgeIdToUse === '{edge_id}') {
              throw new Error('Edge ID not available. Run "Create Edge" test first.')
            }
            response = await edgeApi.update(edgeIdToUse, test.payload as EdgeUpdate)
          }
          break

        case 'DELETE':
          if (endpoint.startsWith('/nodes/')) {
            const nodeId = endpoint.split('/nodes/')[1]
            response = await nodeApi.delete(nodeId)
          } else if (endpoint.startsWith('/edges/')) {
            const edgeIdParam = endpoint.split('/edges/')[1]
            // Use stored edge_id if endpoint has placeholder
            const edgeIdToUse = edgeIdParam === '{edge_id}' && edgeId ? edgeId : edgeIdParam
            if (!edgeIdToUse || edgeIdToUse === '{edge_id}') {
              throw new Error('Edge ID not available. Run "Create Edge" test first.')
            }
            response = await edgeApi.delete(edgeIdToUse)
          }
          break
      }

      const status = response?.status || 200
      const passed = status === (test.expectedStatus || 200)

      return {
        testId: test.id,
        status: passed ? 'passed' : 'failed',
        message: passed ? 'Test passed' : `Expected status ${test.expectedStatus}, got ${status}`,
        response: response?.data,
      }
    } catch (error: any) {
      return {
        testId: test.id,
        status: 'failed',
        message: 'Test failed',
        error: error.response?.data?.detail || error.message || 'Unknown error',
        response: error.response?.data,
      }
    }
  }

  const runAllTests = async () => {
    setIsRunning(true)
    setResults({})
    setEdgeId(null)
    setEdgeIdDisplay(null)
    setCurrentTestId(null)

    // Execute tests sequentially from first to last
    for (let i = 0; i < testCases.length; i++) {
      const test = testCases[i]
      setCurrentTestId(test.id)
      
      // Run the test and wait for it to complete
      const result = await runTest(test)
      updateResult(test.id, result)
      
      // Small delay between tests for better UX
      if (i < testCases.length - 1) {
        await new Promise((resolve) => setTimeout(resolve, 500))
      }
    }

    setCurrentTestId(null)
    setIsRunning(false)
  }

  const runSingleTest = async (test: TestCase) => {
    const result = await runTest(test)
    updateResult(test.id, result)
  }

  const resetTests = () => {
    setResults({})
    setEdgeId(null)
    setEdgeIdDisplay(null)
    setCurrentTestId(null)
  }

  const passedCount = Object.values(results).filter((r) => r.status === 'passed').length
  const failedCount = Object.values(results).filter((r) => r.status === 'failed').length
  const runningCount = Object.values(results).filter((r) => r.status === 'running').length

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Test Runner</h1>
          <p className="text-gray-600 mt-2">
            Execute all API test cases from devforge_test_case.py (runs sequentially from first to last)
          </p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={resetTests}
            className="btn btn-secondary flex items-center space-x-2"
            disabled={isRunning}
          >
            <RefreshCw className="h-5 w-5" />
            <span>Reset</span>
          </button>
          <button
            onClick={runAllTests}
            disabled={isRunning}
            className="btn btn-primary flex items-center space-x-2"
          >
            {isRunning ? (
              <>
                <Loader className="h-5 w-5 animate-spin" />
                <span>Running Tests...</span>
              </>
            ) : (
              <>
                <Play className="h-5 w-5" />
                <span>Run All Tests (Sequential)</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Test Summary */}
      {Object.keys(results).length > 0 && (
        <div className="card">
          <div className="flex items-center space-x-6">
            <div>
              <p className="text-sm text-gray-600">Total Tests</p>
              <p className="text-2xl font-bold text-gray-900">{testCases.length}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Passed</p>
              <p className="text-2xl font-bold text-green-600">{passedCount}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Failed</p>
              <p className="text-2xl font-bold text-red-600">{failedCount}</p>
            </div>
            {(runningCount > 0 || currentTestId) && (
              <div>
                <p className="text-sm text-gray-600">Running</p>
                <p className="text-2xl font-bold text-blue-600">
                  {currentTestId ? testCases.findIndex(t => t.id === currentTestId) + 1 : runningCount}
                </p>
                {currentTestId && (
                  <p className="text-xs text-gray-500 mt-1">
                    Test {currentTestId}
                  </p>
                )}
              </div>
            )}
          </div>
          {edgeIdDisplay && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <p className="text-sm text-gray-600">
                <span className="font-medium">Created Edge ID:</span>{' '}
                <span className="font-mono text-primary-600">{edgeIdDisplay}</span>
              </p>
              <p className="text-xs text-gray-500 mt-1">
                This edge ID is used in tests 6, 7, and 11
              </p>
            </div>
          )}
        </div>
      )}

      {/* Test Cases */}
      <div className="space-y-4">
        {testCases.map((test, index) => {
          const result = results[test.id] || { status: 'pending' as const }
          const isRunning = result.status === 'running' || currentTestId === test.id
          const isCurrent = currentTestId === test.id

          return (
            <div
              key={test.id}
              className={`card border-2 ${
                result.status === 'passed'
                  ? 'border-green-200 bg-green-50'
                  : result.status === 'failed'
                  ? 'border-red-200 bg-red-50'
                  : isRunning || isCurrent
                  ? 'border-blue-200 bg-blue-50'
                  : 'border-gray-200'
              } ${isCurrent ? 'ring-2 ring-blue-400' : ''}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <span className="text-sm font-medium text-gray-500">
                      Test {test.id} ({index + 1}/{testCases.length})
                    </span>
                    <h3 className="text-lg font-semibold text-gray-900">{test.name}</h3>
                    {result.status === 'passed' && (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    )}
                    {result.status === 'failed' && (
                      <XCircle className="h-5 w-5 text-red-600" />
                    )}
                    {(isRunning || isCurrent) && (
                      <Loader className="h-5 w-5 text-blue-600 animate-spin" />
                    )}
                    {isCurrent && (
                      <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded font-medium">
                        Running...
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mb-3">{test.description}</p>
                  <div className="text-xs font-mono text-gray-500 space-y-1">
                    <div>
                      <span className="font-semibold">Method:</span> {test.method}
                    </div>
                    <div>
                      <span className="font-semibold">Endpoint:</span>{' '}
                      {test.endpoint.includes('{edge_id}') && edgeIdDisplay
                        ? test.endpoint.replace('{edge_id}', edgeIdDisplay)
                        : test.endpoint}
                    </div>
                    {test.payload && (
                      <div>
                        <span className="font-semibold">Payload:</span>
                        <pre className="mt-1 bg-gray-100 p-2 rounded overflow-auto">
                          {JSON.stringify(test.payload, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => runSingleTest(test)}
                  disabled={isRunning || isRunning}
                  className="btn btn-secondary ml-4"
                >
                  {isRunning ? 'Running...' : 'Run'}
                </button>
              </div>

              {/* Test Result */}
              {result.status !== 'pending' && result.status !== 'running' && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="space-y-2">
                    <div>
                      <span className="text-sm font-medium text-gray-700">Status: </span>
                      <span
                        className={`text-sm font-semibold ${
                          result.status === 'passed' ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        {result.status.toUpperCase()}
                      </span>
                    </div>
                    {result.message && (
                      <div>
                        <span className="text-sm font-medium text-gray-700">Message: </span>
                        <span className="text-sm text-gray-600">{result.message}</span>
                      </div>
                    )}
                    {result.error && (
                      <div>
                        <span className="text-sm font-medium text-red-700">Error: </span>
                        <span className="text-sm text-red-600">{result.error}</span>
                      </div>
                    )}
                    {result.response && (
                      <details className="mt-2">
                        <summary className="text-sm font-medium text-gray-700 cursor-pointer hover:text-gray-900">
                          View Response
                        </summary>
                        <pre className="mt-2 text-xs bg-gray-100 p-3 rounded overflow-auto max-h-60">
                          {JSON.stringify(result.response, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

