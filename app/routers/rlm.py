"""RLM endpoints: frozen-retriever question answering and experiment viewing.

Why this router builds its own service
--------------------------------------
`app.dependencies.get_service()` points at the production stores (the uploaded
resume and manuals). The RLM experiments run on the synthetic benchmark corpus,
so this router constructs a service per corpus through the benchmark registry
and caches it. The production service is left exactly as the rest of the API
uses it.

Privacy: a corpus holding real personal documents is never sent to a hosted
provider. That check is enforced here as well as in rlm/evaluate.py, because
this endpoint is a second way to reach the same generation call.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]

import benchmark as bench  # noqa: E402
from rlm import loop as rlm_loop  # noqa: E402
from rlm.providers import (  # noqa: E402
    PROVIDERS,
    ProviderUnavailable,
    QuotaExceeded,
    build_provider,
    resolve_key,
)

router = APIRouter(prefix="/rlm", tags=["rlm"])

HOSTED_PROVIDERS = {"anthropic", "google"}
PRIVACY_RESTRICTED_CORPORA = {"production-uploads-v1"}
DEFAULT_CORPUS = "meridian-transit-v1"

_services: Dict[str, Any] = {}
_services_lock = threading.Lock()


def get_corpus_service(corpus_id: str):
    """One service per benchmark corpus, built once."""
    if corpus_id not in _services:
        with _services_lock:
            if corpus_id not in _services:
                try:
                    corpus = bench.resolve_corpus(corpus_id)
                except bench.BenchmarkInvalid as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                _services[corpus_id] = (bench.build_service(corpus), corpus)
    return _services[corpus_id]


def load_dataset_for(corpus_id: str):
    service, corpus = get_corpus_service(corpus_id)
    dataset, provenance = bench.load_dataset(corpus)
    return service, corpus, dataset, provenance


def dataset_size(corpus_id: str) -> Optional[int]:
    """Query count for a corpus, used to judge whether an artifact is complete."""
    try:
        corpus = bench.resolve_corpus(corpus_id)
        rows, _provenance = bench.load_dataset(corpus)
        return len(rows)
    except Exception:
        return None


# ================================= Models ===================================


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    provider: str = "mock"
    model: Optional[str] = None
    corpus_id: str = DEFAULT_CORPUS
    top_k: int = 5
    max_steps: int = 3
    max_retrievals: int = 3
    temperature: float = 0.0
    max_tokens: int = 512
    query_id: Optional[int] = None


class EvidenceChunk(BaseModel):
    node_id: str
    text: str
    hop: int
    retrieved_by_vector: bool


class AskResponse(BaseModel):
    question: str
    answer: str
    provider: str
    model: str
    evaluative: bool
    stop_reason: str
    retrieval_calls: int
    generation_calls: int
    input_tokens: int
    output_tokens: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    steps: List[Dict[str, Any]]
    evidence: List[EvidenceChunk]
    retrieval: Dict[str, Any]
    answer_evaluation: Optional[Dict[str, Any]] = None
    gold_answer: Optional[Dict[str, Any]] = None


# ================================ Endpoints =================================


@router.get("/status")
def status() -> Dict[str, Any]:
    """Frozen retriever, provider readiness and corpora, without spending quota.

    Provider readiness is reported from credential presence plus a model
    listing, which does not consume generation quota. It deliberately does not
    make a generation call, so this endpoint is safe to poll.
    """
    try:
        frozen = rlm_loop.load_frozen_retrieval_config()
        retrieval = {
            "available": True,
            "variant": frozen["selected"]["variant"],
            "description": frozen["selected"]["description"],
            "alpha": frozen["selected"]["alpha"],
            "beta": frozen["selected"]["beta"],
            "gamma": frozen["selected"]["gamma"],
            "seed_k": frozen["selected"]["seed_k"],
            "max_depth": frozen["selected"]["max_depth"],
            "production_service_modified": frozen["selected"]["production_service_modified"],
            "control_metrics": frozen["control"]["metrics"],
            "selected_metrics": frozen["selected_metrics"],
            "corpus_fingerprint": frozen["provenance"]["corpus_fingerprint"],
            "dataset_sha256": frozen["provenance"]["dataset_sha256"],
        }
    except Exception as exc:
        retrieval = {"available": False, "detail": str(exc)}

    providers: List[Dict[str, Any]] = []
    for name in sorted(PROVIDERS):
        entry: Dict[str, Any] = {"name": name, "hosted": name in HOSTED_PROVIDERS}
        if name == "mock":
            entry.update({"ready": True, "key_present": True, "models": ["mock-deterministic-1"],
                          "evaluative": False})
            providers.append(entry)
            continue
        key_name = "ANTHROPIC_API_KEY" if name == "anthropic" else "GOOGLE_API_KEY"
        entry["key_present"] = bool(resolve_key(key_name))
        entry["evaluative"] = True
        if not entry["key_present"]:
            entry.update({"ready": False, "detail": f"{key_name} is not set", "models": []})
            providers.append(entry)
            continue
        try:
            models = build_provider(name).list_models()
            entry.update({"ready": True, "models": sorted(models)[:60], "model_count": len(models)})
        except Exception as exc:
            entry.update({"ready": False, "detail": f"{type(exc).__name__}: {exc}"[:300], "models": []})
        providers.append(entry)

    corpora = []
    for corpus_id, entry in bench.CORPUS_REGISTRY.items():
        corpora.append({
            "corpus_id": corpus_id,
            "description": entry.get("description"),
            "has_dataset": bool(entry.get("dataset")),
            "privacy_restricted": corpus_id in PRIVACY_RESTRICTED_CORPORA,
            "vector_doc_count": entry.get("vector_doc_count"),
        })

    return {
        "retrieval": retrieval,
        "providers": providers,
        "corpora": corpora,
        "loop": {
            "version": rlm_loop.LOOP_VERSION,
            "prompt_template_sha256": rlm_loop.TEMPLATE_HASHES,
        },
    }


@router.get("/dataset")
def dataset(corpus_id: str = DEFAULT_CORPUS, limit: int = 60) -> Dict[str, Any]:
    """Benchmark questions with their labels, so the UI can offer real examples."""
    _service, corpus, rows, provenance = load_dataset_for(corpus_id)
    return {
        "corpus_id": corpus["corpus_id"],
        "dataset": provenance["dataset_path"],
        "dataset_version": provenance["dataset_version"],
        "dataset_sha256": provenance["dataset_sha256"],
        "count": len(rows),
        "queries": [
            {
                "id": item.get("id"),
                "query": item.get("query"),
                "category": item.get("category"),
                "hops": item.get("hops"),
                "relevant": item.get("relevant"),
                "gold_answer": item.get("gold_answer"),
            }
            for item in rows[:limit]
        ],
    }


@router.get("/experiments")
def experiments() -> Dict[str, Any]:
    """Saved experiment artifacts, summarised. Completeness is reported, not hidden."""
    rows: List[Dict[str, Any]] = []
    for path in sorted(ROOT.glob("experiment_rlm_eval_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rows.append({"file": path.name, "error": f"unreadable: {exc}"})
            continue
        reproducibility = payload.get("reproducibility", {})
        generation = reproducibility.get("generation", {})

        # Artifacts written before completeness was recorded can claim
        # "valid" while holding a fraction of the dataset. Rather than edit a
        # saved artifact, derive the missing fields here and say they were
        # derived, so the UI cannot present a 1-of-60 run as a full one.
        evaluated = payload.get("queries_evaluated", len(payload.get("per_query", [])))
        in_dataset = payload.get("queries_in_dataset")
        derived = False
        if in_dataset is None:
            corpus_id = reproducibility.get("corpus_id")
            try:
                in_dataset = dataset_size(corpus_id) if corpus_id else None
                derived = in_dataset is not None
            except Exception:
                in_dataset = None
        complete = payload.get("complete")
        status = payload.get("status")
        if complete is None and in_dataset:
            complete = evaluated >= in_dataset
            if not complete and status == "valid":
                status = "partial"

        rows.append({
            "file": path.name,
            "status": status,
            "status_derived": derived,
            "recorded_status": payload.get("status"),
            "evaluative": payload.get("evaluative"),
            "complete": complete,
            "queries_evaluated": evaluated,
            "queries_in_dataset": in_dataset,
            "quota_stop": payload.get("quota_stop"),
            "caveat": payload.get("caveat") or payload.get("partial_caveat"),
            "provider": generation.get("provider"),
            "model": generation.get("model"),
            "corpus_id": reproducibility.get("corpus_id"),
            "retrieval_variant": (reproducibility.get("retrieval") or {}).get("variant"),
            "recorded_at": reproducibility.get("recorded_at"),
            "aggregate": payload.get("aggregate"),
        })
    return {"count": len(rows), "experiments": rows}


@router.get("/experiments/{file_name}")
def experiment_detail(file_name: str) -> Dict[str, Any]:
    """One artifact in full. The name is matched against the known glob only."""
    allowed = {path.name for path in ROOT.glob("experiment_rlm_eval_*.json")}
    if file_name not in allowed:
        raise HTTPException(status_code=404, detail=f"No such experiment artifact: {file_name}")
    return json.loads((ROOT / file_name).read_text(encoding="utf-8"))


@router.get("/ablation")
def ablation(corpus_id: str = DEFAULT_CORPUS) -> Dict[str, Any]:
    """The fusion ablation and the frozen selection, for the comparison view."""
    ablation_path = ROOT / f"experiment_fusion_ablation_{corpus_id}.json"
    frozen_path = ROOT / "experiment_frozen_retrieval_config.json"
    if not ablation_path.exists():
        raise HTTPException(status_code=404, detail=f"No ablation artifact for {corpus_id}")
    payload = json.loads(ablation_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else None
    return {
        "corpus_id": corpus_id,
        "configuration": payload.get("configuration"),
        "control_faithfulness": payload.get("control_faithfulness"),
        "variant_descriptions": payload.get("variant_descriptions"),
        "variants": {
            name: {
                "aggregate": run["aggregate"],
                "by_category": run["by_category"],
                "seed_k": run["seed_k"],
            }
            for name, run in payload.get("runs", {}).items()
        },
        "comparisons": payload.get("comparisons"),
        "frozen": None if frozen is None else {
            "selected": frozen["selected"],
            "selection_rule": frozen["selection_rule"],
            "decisions": frozen["decisions"],
            "trade_offs": frozen["trade_offs"],
        },
    }


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Run the bounded loop for one question against the frozen retriever.

    Failures are reported as themselves: a quota stop returns 429 with the
    provider's own message, a missing credential returns 400. Neither is turned
    into an empty answer.
    """
    if request.corpus_id in PRIVACY_RESTRICTED_CORPORA and request.provider in HOSTED_PROVIDERS:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Corpus {request.corpus_id!r} holds real personal documents and is never sent "
                f"to the hosted provider {request.provider!r}. Use the synthetic corpus."
            ),
        )

    service, _corpus = get_corpus_service(request.corpus_id)
    try:
        frozen = rlm_loop.load_frozen_retrieval_config()
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    retriever = rlm_loop.FrozenRetriever(service, frozen)

    try:
        provider = build_provider(
            request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except ProviderUnavailable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.provider in HOSTED_PROVIDERS:
        try:
            provider.verify_model()
        except ProviderUnavailable as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    bounds = rlm_loop.LoopBounds(
        max_steps=request.max_steps,
        max_retrievals=request.max_retrievals,
        top_k=request.top_k,
    )

    try:
        run = rlm_loop.run_loop(retriever, provider, request.question, bounds)
    except QuotaExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "provider": exc.provider,
                "message": exc.detail[:600],
                "hint": (
                    "Generation quota is exhausted for this key. The mock provider still works "
                    "for plumbing, and saved experiment artifacts remain viewable."
                ),
            },
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    evidence_text = {chunk["node_id"]: chunk for chunk in retriever.retrieve(request.question, request.top_k)}
    evidence: List[EvidenceChunk] = []
    for node_id in run.evidence_ids:
        chunk = evidence_text.get(node_id)
        if chunk is None:
            node = service.graph_db.get_node(node_id)
            chunk = {
                "node_id": node_id,
                "text": node.text if node else "",
                "hop": -1,
                "retrieved_by_vector": False,
            }
        evidence.append(EvidenceChunk(**chunk))

    answer_evaluation = None
    gold_answer = None
    if request.query_id is not None:
        _s, _c, rows, _p = load_dataset_for(request.corpus_id)
        match = next((item for item in rows if int(item.get("id", -1)) == request.query_id), None)
        if match and match.get("gold_answer"):
            from rlm.evaluate import score_answer, station_universe

            gold_answer = match["gold_answer"]
            answer_evaluation = score_answer(run.answer, gold_answer, station_universe(service))

    record = run.as_record()
    return AskResponse(
        question=request.question,
        answer=run.answer,
        provider=provider.config.provider,
        model=provider.config.model,
        evaluative=request.provider != "mock",
        stop_reason=run.stop_reason,
        retrieval_calls=run.retrieval_calls,
        generation_calls=run.generation_calls,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        retrieval_latency_ms=record["retrieval_latency_ms"],
        generation_latency_ms=record["generation_latency_ms"],
        total_latency_ms=record["total_latency_ms"],
        steps=run.steps,
        evidence=evidence,
        retrieval={
            "variant": retriever.variant_name,
            "weights": retriever.weights,
            "seed_k": retriever.seed_k,
            "max_depth": retriever.max_depth,
            "top_k": bounds.top_k,
        },
        answer_evaluation=answer_evaluation,
        gold_answer=gold_answer,
    )
