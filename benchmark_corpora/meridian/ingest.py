"""Ingest the Meridian Transit benchmark corpus into its own dedicated stores.

Writes only:
    db/chroma_db_bench_meridian/        (vector index)
    db/graph_data_bench_meridian.json   (graph)

It never writes to db/chroma_db_service, db/graph_data_service.json,
vector_db/db/chroma_db_with_metadata or db/graph_data.json, so the production
stores and the Romeo and Juliet baseline corpus are left exactly as they are.

Run:
    python benchmark_corpora/meridian/ingest.py          # refuses if stores exist
    python benchmark_corpora/meridian/ingest.py --force  # rebuild from scratch
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VECTOR_PATH = ROOT / "db" / "chroma_db_bench_meridian"
GRAPH_PATH = ROOT / "db" / "graph_data_bench_meridian.json"

PROTECTED = {
    ROOT / "db" / "chroma_db_service",
    ROOT / "db" / "graph_data_service.json",
    ROOT / "vector_db" / "db" / "chroma_db_with_metadata",
    ROOT / "db" / "graph_data.json",
}


def main() -> int:
    force = "--force" in sys.argv

    assert VECTOR_PATH not in PROTECTED and GRAPH_PATH not in PROTECTED, "refusing to target a protected store"

    documents = json.loads((HERE / "corpus_documents.json").read_text(encoding="utf-8"))
    graph_payload = json.loads((HERE / "corpus_graph.json").read_text(encoding="utf-8"))

    if (VECTOR_PATH.exists() or GRAPH_PATH.exists()) and not force:
        print(f"Stores already exist:\n  {VECTOR_PATH}\n  {GRAPH_PATH}\nPass --force to rebuild.")
        return 1

    if force:
        if VECTOR_PATH.exists():
            shutil.rmtree(VECTOR_PATH)
        if GRAPH_PATH.exists():
            GRAPH_PATH.unlink()

    VECTOR_PATH.mkdir(parents=True, exist_ok=True)

    # --- Graph: the generated file already uses the graph_db JSON format. ---
    GRAPH_PATH.write_text(json.dumps(graph_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    from graph_db.graph_db import GraphDatabase

    graph_db = GraphDatabase(db_path=str(GRAPH_PATH), auto_persist=True)
    assert graph_db.graph.number_of_nodes() == len(graph_payload["nodes"]), "graph node count mismatch after load"
    assert graph_db.graph.number_of_edges() == len(graph_payload["edges"]), "graph edge count mismatch after load"
    print(f"graph: {graph_db.graph.number_of_nodes()} nodes, {graph_db.graph.number_of_edges()} edges -> {GRAPH_PATH}")

    # --- Vector index: same VectorDatabase class the service uses. ---
    from app.vector_db import VectorDatabase

    vector_db = VectorDatabase(persist_directory=str(VECTOR_PATH))
    ids = [doc["id"] for doc in documents]
    texts = [doc["text"] for doc in documents]
    metadatas = []
    for doc in documents:
        flat = {}
        for key, value in doc["metadata"].items():
            flat[key] = ", ".join(map(str, value)) if isinstance(value, list) else value
        metadatas.append(flat)

    batch = 50
    for start in range(0, len(ids), batch):
        vector_db.add_documents(ids[start:start + batch], texts[start:start + batch], metadatas[start:start + batch])
        print(f"  embedded {min(start + batch, len(ids))}/{len(ids)}")

    stored = vector_db.db.get(include=["metadatas"])
    stored_ids = {(meta or {}).get("id") for meta in (stored.get("metadatas") or [])}
    missing = sorted(set(ids) - stored_ids)
    assert not missing, f"documents missing from the vector index: {missing[:10]}"
    print(f"vector: {len(stored.get('ids') or [])} documents -> {VECTOR_PATH}")

    for path in sorted(PROTECTED):
        print(f"untouched: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
