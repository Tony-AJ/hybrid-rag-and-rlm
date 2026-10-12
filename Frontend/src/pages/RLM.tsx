import { useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  Brain, Play, AlertTriangle, CheckCircle, XCircle, Loader, Database,
  FileBarChart, Layers, Info,
} from 'lucide-react'
import { rlmApi, type RlmAskResponse, type RlmDatasetQuery } from '../services/api'

type Tab = 'ask' | 'experiments' | 'ablation'

const CORPUS = 'meridian-transit-v1'

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="text-xs uppercase tracking-wide text-gray-500">{label}</div>
      <div className="text-2xl font-semibold text-gray-900 mt-1">{value}</div>
      {hint && <div className="text-xs text-gray-500 mt-1">{hint}</div>}
    </div>
  )
}

/** Quota and credential problems are shown as themselves, never as an empty answer. */
function ErrorPanel({ error }: { error: any }) {
  const detail = error?.response?.data?.detail
  const quota = detail && typeof detail === 'object' && detail.error === 'quota_exceeded'
  const status = error?.response?.status

  return (
    <div className={`rounded-lg border p-4 ${quota ? 'border-amber-300 bg-amber-50' : 'border-red-300 bg-red-50'}`}>
      <div className="flex items-start space-x-3">
        <AlertTriangle className={`h-5 w-5 mt-0.5 ${quota ? 'text-amber-600' : 'text-red-600'}`} />
        <div className="min-w-0">
          <div className={`font-medium ${quota ? 'text-amber-900' : 'text-red-900'}`}>
            {quota ? `${detail.provider} quota exhausted` : `Request failed${status ? ` (HTTP ${status})` : ''}`}
          </div>
          <div className={`text-sm mt-1 break-words ${quota ? 'text-amber-800' : 'text-red-800'}`}>
            {quota ? detail.message : (typeof detail === 'string' ? detail : error?.message || 'Unknown error')}
          </div>
          {quota && <div className="text-sm text-amber-700 mt-2">{detail.hint}</div>}
        </div>
      </div>
    </div>
  )
}

function AnswerView({ result }: { result: RlmAskResponse }) {
  const evaluation = result.answer_evaluation
  return (
    <div className="space-y-4">
      {!result.evaluative && (
        <div className="rounded-lg border border-blue-300 bg-blue-50 p-3 flex items-start space-x-2">
          <Info className="h-4 w-4 text-blue-600 mt-0.5" />
          <span className="text-sm text-blue-800">
            Mock provider: no reasoning is performed. This exercises retrieval, the loop and the
            accounting only, so the answer is not a quality signal.
          </span>
        </div>
      )}

      <div className="bg-white rounded-lg border border-gray-200 p-5">
        <div className="text-xs uppercase tracking-wide text-gray-500 mb-2">Answer</div>
        <p className="text-gray-900 whitespace-pre-wrap">{result.answer}</p>
        <div className="text-xs text-gray-500 mt-3">
          {result.provider} / {result.model} · stopped: {result.stop_reason}
        </div>
      </div>

      {evaluation && (
        <div className={`rounded-lg border p-4 ${evaluation.correct ? 'border-green-300 bg-green-50' : 'border-orange-300 bg-orange-50'}`}>
          <div className="flex items-center space-x-2">
            {evaluation.correct
              ? <CheckCircle className="h-5 w-5 text-green-600" />
              : <XCircle className="h-5 w-5 text-orange-600" />}
            <span className="font-medium text-gray-900">
              {evaluation.correct ? 'Matches gold answer' : 'Does not match gold answer'}
            </span>
            <span className="text-xs text-gray-600">({evaluation.kind})</span>
          </div>
          {evaluation.kind === 'entity_set' && (
            <div className="text-sm text-gray-700 mt-2 space-y-1">
              <div>expected: <code className="bg-white px-1 rounded">{(evaluation.expected || []).join(', ')}</code></div>
              <div>extracted: <code className="bg-white px-1 rounded">{(evaluation.extracted || []).join(', ') || '(none)'}</code></div>
              <div>precision {evaluation.precision} · recall {evaluation.recall}</div>
            </div>
          )}
          {evaluation.kind === 'predicates' && (
            <div className="text-sm text-gray-700 mt-2">
              {Object.entries(evaluation.predicates || {}).map(([key, ok]) => (
                <div key={key} className="flex items-center space-x-2">
                  {ok ? <CheckCircle className="h-3 w-3 text-green-600" /> : <XCircle className="h-3 w-3 text-orange-600" />}
                  <span>{key}</span>
                </div>
              ))}
              <div className="mt-1">{evaluation.satisfied} of {evaluation.total} satisfied</div>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Metric label="retrieval calls" value={String(result.retrieval_calls)} />
        <Metric label="generation calls" value={String(result.generation_calls)} />
        <Metric label="tokens in / out" value={`${result.input_tokens} / ${result.output_tokens}`} />
        <Metric label="latency" value={`${result.total_latency_ms.toFixed(0)} ms`}
          hint={`retrieval ${result.retrieval_latency_ms.toFixed(0)} · generation ${result.generation_latency_ms.toFixed(0)}`} />
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-5">
        <div className="text-xs uppercase tracking-wide text-gray-500 mb-3">Loop steps</div>
        <div className="space-y-3">
          {result.steps.map((step) => (
            <div key={step.step} className="border-l-2 border-primary-300 pl-3">
              <div className="text-sm font-medium text-gray-900">Step {step.step}</div>
              <div className="text-sm text-gray-700">query: {step.query}</div>
              <div className="text-xs text-gray-500 mt-1">
                retrieved: {step.retrieved.join(', ') || '(none)'}
              </div>
              {step.new_evidence?.length > 0 && (
                <div className="text-xs text-gray-500">new evidence: {step.new_evidence.join(', ')}</div>
              )}
              {step.decision && <div className="text-xs text-gray-600 mt-1">decision: {step.decision}</div>}
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-5">
        <div className="text-xs uppercase tracking-wide text-gray-500 mb-3">
          Evidence ({result.evidence.length} chunks) · retriever {result.retrieval.variant}
        </div>
        <div className="space-y-2">
          {result.evidence.map((chunk) => (
            <div key={chunk.node_id} className="border border-gray-100 rounded p-3">
              <div className="flex items-center justify-between">
                <code className="text-xs font-medium text-primary-700">{chunk.node_id}</code>
                <div className="flex items-center space-x-2">
                  <span className={`text-xs px-2 py-0.5 rounded ${chunk.retrieved_by_vector ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800'}`}>
                    {chunk.retrieved_by_vector ? 'vector' : 'graph-only'}
                  </span>
                  <span className="text-xs text-gray-500">hop {chunk.hop}</span>
                </div>
              </div>
              <p className="text-sm text-gray-700 mt-2 line-clamp-3">{chunk.text}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function RLM() {
  const [tab, setTab] = useState<Tab>('ask')
  const [question, setQuestion] = useState('')
  const [queryId, setQueryId] = useState<number | undefined>(undefined)
  const [provider, setProvider] = useState('mock')
  const [model, setModel] = useState('')
  const [topK, setTopK] = useState(5)
  const [maxSteps, setMaxSteps] = useState(3)
  const [maxRetrievals, setMaxRetrievals] = useState(3)
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null)

  const statusQuery = useQuery({ queryKey: ['rlm-status'], queryFn: () => rlmApi.status().then(r => r.data) })
  const datasetQuery = useQuery({ queryKey: ['rlm-dataset'], queryFn: () => rlmApi.dataset(CORPUS).then(r => r.data) })
  const experimentsQuery = useQuery({
    queryKey: ['rlm-experiments'],
    queryFn: () => rlmApi.experiments().then(r => r.data),
    enabled: tab === 'experiments',
  })
  const ablationQuery = useQuery({
    queryKey: ['rlm-ablation'],
    queryFn: () => rlmApi.ablation(CORPUS).then(r => r.data),
    enabled: tab === 'ablation',
  })
  const detailQuery = useQuery({
    queryKey: ['rlm-experiment', selectedExperiment],
    queryFn: () => rlmApi.experimentDetail(selectedExperiment!).then(r => r.data),
    enabled: !!selectedExperiment,
  })

  const askMutation = useMutation({
    mutationFn: () => rlmApi.ask({
      question, provider, model: model || undefined, corpus_id: CORPUS,
      top_k: topK, max_steps: maxSteps, max_retrievals: maxRetrievals, query_id: queryId,
    }).then(r => r.data),
  })

  const status = statusQuery.data
  const providers = status?.providers ?? []
  const activeProvider = providers.find(p => p.name === provider)
  const retrieval = status?.retrieval

  const grouped = useMemo(() => {
    const map = new Map<string, RlmDatasetQuery[]>()
    for (const row of datasetQuery.data?.queries ?? []) {
      const list = map.get(row.category) ?? []
      list.push(row)
      map.set(row.category, list)
    }
    return map
  }, [datasetQuery.data])

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <Brain className="h-7 w-7 text-primary-600" />
        <div>
          <h2 className="text-2xl font-bold text-gray-900">RLM</h2>
          <p className="text-sm text-gray-500">
            Bounded retrieve-and-reason loop over the frozen retriever
          </p>
        </div>
      </div>

      {/* Frozen retriever + provider readiness */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <div className="flex items-center space-x-2 mb-3">
            <Layers className="h-4 w-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-900">Frozen retriever</span>
          </div>
          {retrieval?.available ? (
            <div className="text-sm text-gray-700 space-y-1">
              <div>variant <code className="bg-gray-100 px-1 rounded">{retrieval.variant}</code></div>
              <div className="text-gray-600">{retrieval.description}</div>
              <div>alpha {retrieval.alpha} · beta {retrieval.beta} · gamma {retrieval.gamma} · depth {retrieval.max_depth}</div>
              {retrieval.selected_metrics && retrieval.control_metrics && (
                <div className="text-xs text-gray-600 pt-2">
                  ndcg@5 {retrieval.selected_metrics['ndcg@5']} vs control {retrieval.control_metrics['ndcg@5']} ·
                  recall@5 {retrieval.selected_metrics['recall@5']} vs control {retrieval.control_metrics['recall@5']}
                </div>
              )}
              <div className="text-xs text-gray-500">
                production service modified: {String(retrieval.production_service_modified)}
              </div>
            </div>
          ) : (
            <div className="text-sm text-amber-700">{retrieval?.detail ?? 'loading'}</div>
          )}
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <div className="flex items-center space-x-2 mb-3">
            <Database className="h-4 w-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-900">Providers</span>
          </div>
          <div className="space-y-2">
            {providers.map(p => (
              <div key={p.name} className="flex items-start justify-between text-sm">
                <div className="min-w-0">
                  <span className="font-medium text-gray-900">{p.name}</span>
                  {!p.evaluative && <span className="text-xs text-gray-500 ml-2">non-evaluative</span>}
                  {p.detail && <div className="text-xs text-gray-500 break-words max-w-md">{p.detail}</div>}
                  {p.ready && p.model_count && (
                    <div className="text-xs text-gray-500">{p.model_count} models reachable</div>
                  )}
                </div>
                <span className={`text-xs px-2 py-0.5 rounded whitespace-nowrap ${p.ready ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'}`}>
                  {p.ready ? 'ready' : 'unavailable'}
                </span>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-3">
            Readiness is checked by listing models, which does not consume generation quota.
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex space-x-2 border-b border-gray-200">
        {([['ask', 'Ask'], ['experiments', 'Experiments'], ['ablation', 'Fusion ablation']] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${tab === key ? 'border-primary-600 text-primary-700' : 'border-transparent text-gray-600 hover:text-gray-900'}`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'ask' && (
        <div className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-5 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Question</label>
              <textarea
                value={question}
                onChange={(e) => { setQuestion(e.target.value); setQueryId(undefined) }}
                rows={2}
                placeholder="Which maintenance procedures cover the fault that stopped service at Harbour Row?"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              />
            </div>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <div>
                <label className="block text-xs text-gray-600 mb-1">Provider</label>
                <select value={provider} onChange={(e) => setProvider(e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-2 py-2 text-sm">
                  {providers.map(p => (
                    <option key={p.name} value={p.name} disabled={!p.ready}>
                      {p.name}{p.ready ? '' : ' (unavailable)'}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-600 mb-1">Model</label>
                <select value={model} onChange={(e) => setModel(e.target.value)}
                  className="w-full border border-gray-300 rounded-lg px-2 py-2 text-sm">
                  <option value="">default</option>
                  {(activeProvider?.models ?? []).map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-600 mb-1">top_k</label>
                <input type="number" min={1} max={20} value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full border border-gray-300 rounded-lg px-2 py-2 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-600 mb-1">max steps</label>
                <input type="number" min={1} max={6} value={maxSteps}
                  onChange={(e) => setMaxSteps(Number(e.target.value))}
                  className="w-full border border-gray-300 rounded-lg px-2 py-2 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-600 mb-1">max retrievals</label>
                <input type="number" min={1} max={6} value={maxRetrievals}
                  onChange={(e) => setMaxRetrievals(Number(e.target.value))}
                  className="w-full border border-gray-300 rounded-lg px-2 py-2 text-sm" />
              </div>
            </div>

            <button
              onClick={() => askMutation.mutate()}
              disabled={!question.trim() || askMutation.isPending || !activeProvider?.ready}
              className="flex items-center space-x-2 bg-primary-600 text-white px-4 py-2 rounded-lg text-sm font-medium disabled:bg-gray-300"
            >
              {askMutation.isPending ? <Loader className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              <span>{askMutation.isPending ? 'Running loop' : 'Run loop'}</span>
            </button>

            {queryId !== undefined && (
              <p className="text-xs text-gray-500">
                Benchmark query #{queryId} selected: the answer will be scored against its gold answer.
              </p>
            )}
          </div>

          {askMutation.isError && <ErrorPanel error={askMutation.error} />}
          {askMutation.data && <AnswerView result={askMutation.data} />}

          {/* Benchmark questions, grouped by family */}
          <div className="bg-white rounded-lg border border-gray-200 p-5">
            <div className="text-sm font-medium text-gray-900 mb-1">
              Benchmark questions ({datasetQuery.data?.count ?? 0})
            </div>
            <p className="text-xs text-gray-500 mb-3">
              Dataset {datasetQuery.data?.dataset_version} · picking one scores the answer against its gold answer.
            </p>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {[...grouped.entries()].map(([category, rows]) => (
                <div key={category}>
                  <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">
                    {category} ({rows.length})
                  </div>
                  <div className="space-y-1">
                    {rows.map(row => (
                      <button
                        key={row.id}
                        onClick={() => { setQuestion(row.query); setQueryId(row.id) }}
                        className={`block w-full text-left text-sm px-3 py-2 rounded hover:bg-gray-50 ${queryId === row.id ? 'bg-primary-50 text-primary-800' : 'text-gray-700'}`}
                      >
                        <span className="text-gray-400 mr-2">#{row.id}</span>{row.query}
                        <span className="text-xs text-gray-400 ml-2">{row.hops} hop</span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {tab === 'experiments' && (
        <div className="space-y-4">
          {experimentsQuery.isLoading && <div className="text-sm text-gray-500">loading</div>}
          {experimentsQuery.isError && <ErrorPanel error={experimentsQuery.error} />}
          {(experimentsQuery.data?.experiments ?? []).map(exp => (
            <div key={exp.file} className="bg-white rounded-lg border border-gray-200 p-5">
              <div className="flex items-start justify-between">
                <div className="min-w-0">
                  <div className="flex items-center space-x-2">
                    <FileBarChart className="h-4 w-4 text-gray-500" />
                    <code className="text-sm font-medium text-gray-900">{exp.file}</code>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {exp.provider} / {exp.model} · {exp.corpus_id} · retriever {exp.retrieval_variant} · {exp.recorded_at}
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <span className={`text-xs px-2 py-0.5 rounded whitespace-nowrap ${
                    exp.status === 'valid' ? 'bg-green-100 text-green-800'
                      : exp.status === 'partial' ? 'bg-amber-100 text-amber-800'
                      : 'bg-gray-100 text-gray-700'}`}>
                    {exp.status}
                  </span>
                  {exp.status_derived && (
                    <div className="text-xs text-amber-700 mt-1 max-w-[12rem]">
                      derived: artifact recorded "{exp.recorded_status}" without a query count
                    </div>
                  )}
                </div>
              </div>

              {exp.caveat && (
                <div className="mt-3 text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded p-3">
                  {exp.caveat}
                </div>
              )}
              {exp.quota_stop && (
                <div className="mt-2 text-xs text-amber-700">
                  stopped by {exp.quota_stop.provider} quota after {exp.quota_stop.completed} queries,
                  {' '}{exp.quota_stop.remaining} remaining
                </div>
              )}

              {exp.aggregate && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4">
                  <Metric label="queries" value={`${exp.queries_evaluated ?? '?'}/${exp.queries_in_dataset ?? '?'}`} />
                  <Metric label="answer accuracy"
                    value={exp.evaluative ? String(exp.aggregate.answer_level.accuracy ?? '-') : 'n/a'}
                    hint={exp.evaluative ? undefined : 'mock: not a quality signal'} />
                  <Metric label="evidence coverage" value={String(exp.aggregate.chunk_level.evidence_coverage)} />
                  <Metric label="retrieval calls" value={String(exp.aggregate.cost.mean_retrieval_calls)} />
                  <Metric label="tokens in/out"
                    value={`${exp.aggregate.cost.mean_input_tokens}/${exp.aggregate.cost.mean_output_tokens}`} />
                </div>
              )}

              <button
                onClick={() => setSelectedExperiment(selectedExperiment === exp.file ? null : exp.file)}
                className="text-sm text-primary-700 mt-3"
              >
                {selectedExperiment === exp.file ? 'hide per-query detail' : 'show per-query detail'}
              </button>

              {selectedExperiment === exp.file && detailQuery.data && (
                <div className="mt-3 border-t border-gray-100 pt-3">
                  <div className="text-xs uppercase tracking-wide text-gray-500 mb-2">By category</div>
                  <table className="w-full text-sm mb-4">
                    <thead>
                      <tr className="text-left text-xs text-gray-500">
                        <th className="py-1">category</th><th>n</th><th>accuracy</th><th>coverage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(detailQuery.data.by_category ?? {}).map(([cat, b]: [string, any]) => (
                        <tr key={cat} className="border-t border-gray-100">
                          <td className="py-1 text-gray-700">{cat}</td>
                          <td className="text-gray-700">{b.queries}</td>
                          <td className="text-gray-700">{b.accuracy}</td>
                          <td className="text-gray-700">{b.coverage}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  <div className="text-xs uppercase tracking-wide text-gray-500 mb-2">Per query</div>
                  <div className="space-y-2 max-h-80 overflow-y-auto">
                    {(detailQuery.data.per_query ?? []).map((row: any) => (
                      <div key={row.id} className="border border-gray-100 rounded p-3">
                        <div className="flex items-start justify-between">
                          <div className="text-sm text-gray-900">
                            <span className="text-gray-400 mr-2">#{row.id}</span>{row.question}
                          </div>
                          {row.answer_evaluation?.correct
                            ? <CheckCircle className="h-4 w-4 text-green-600 shrink-0" />
                            : <XCircle className="h-4 w-4 text-orange-500 shrink-0" />}
                        </div>
                        <div className="text-sm text-gray-700 mt-1">{row.answer}</div>
                        <div className="text-xs text-gray-500 mt-1">
                          coverage {row.evidence_coverage?.coverage} · calls {row.cost?.retrieval_calls} ·
                          {' '}tokens {row.cost?.input_tokens}/{row.cost?.output_tokens} · {row.stop_reason}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {tab === 'ablation' && (
        <div className="space-y-4">
          {ablationQuery.isLoading && <div className="text-sm text-gray-500">loading</div>}
          {ablationQuery.isError && <ErrorPanel error={ablationQuery.error} />}
          {ablationQuery.data && (
            <>
              <div className="bg-white rounded-lg border border-gray-200 p-5">
                <div className="text-sm font-medium text-gray-900 mb-2">Selection</div>
                <div className="text-sm text-gray-700">
                  frozen variant <code className="bg-gray-100 px-1 rounded">{ablationQuery.data.frozen?.selected.variant}</code>
                  {' '}· rule: primary {ablationQuery.data.frozen?.selection_rule.primary_metric},
                  {' '}margin {ablationQuery.data.frozen?.selection_rule.complexity_margin_on_primary},
                  {' '}guardrail {ablationQuery.data.frozen?.selection_rule.guardrail_category}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  control reproduces production on {ablationQuery.data.control_faithfulness?.queries_checked} queries,
                  {' '}{ablationQuery.data.control_faithfulness?.mismatches} mismatches
                </div>
              </div>

              <div className="bg-white rounded-lg border border-gray-200 p-5 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-gray-500">
                      <th className="py-2">variant</th><th>seed_k</th><th>precision@5</th>
                      <th>recall@5</th><th>mrr</th><th>ndcg@5</th><th>graph in top-5</th><th>ms</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(ablationQuery.data.variants ?? {}).map(([name, run]: [string, any]) => {
                      const selected = name === ablationQuery.data.frozen?.selected.variant
                      return (
                        <tr key={name} className={`border-t border-gray-100 ${selected ? 'bg-primary-50' : ''}`}>
                          <td className="py-2 font-medium text-gray-900">
                            {name}{selected && <span className="text-xs text-primary-700 ml-2">frozen</span>}
                          </td>
                          <td className="text-gray-700">{run.seed_k}</td>
                          <td className="text-gray-700">{run.aggregate['precision@5']}</td>
                          <td className="text-gray-700">{run.aggregate['recall@5']}</td>
                          <td className="text-gray-700">{run.aggregate['mrr']}</td>
                          <td className="text-gray-700 font-medium">{run.aggregate['ndcg@5']}</td>
                          <td className="text-gray-700">{run.aggregate['graph_only_in_top_k']}</td>
                          <td className="text-gray-500">{run.aggregate['mean_latency_ms']?.toFixed?.(1)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              <div className="bg-white rounded-lg border border-gray-200 p-5">
                <div className="text-sm font-medium text-gray-900 mb-2">Decisions</div>
                <div className="space-y-1">
                  {(ablationQuery.data.frozen?.decisions ?? []).map((d: any) => (
                    <div key={d.variant} className="text-sm flex items-start space-x-2">
                      <span className={`text-xs px-2 py-0.5 rounded whitespace-nowrap ${d.adopted ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'}`}>
                        {d.adopted ? 'adopt' : 'reject'}
                      </span>
                      <span className="text-gray-900">{d.variant}</span>
                      <span className="text-gray-500 text-xs">vs {d.base} · {d.reason}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white rounded-lg border border-gray-200 p-5">
                <div className="text-sm font-medium text-gray-900 mb-2">Recorded trade-offs</div>
                <div className="space-y-2">
                  {(ablationQuery.data.frozen?.trade_offs ?? []).map((note: any, index: number) => (
                    <div key={index} className="text-sm text-gray-700">
                      <span className="text-xs uppercase tracking-wide text-gray-500 mr-2">{note.kind}</span>
                      {note.kind === 'regression'
                        ? `${note.category}: ${note.control} -> ${note.selected} (${note.query_equivalents} query-equivalents)`
                        : note.detail}
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
