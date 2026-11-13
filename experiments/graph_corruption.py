"""In-memory graph corruption for robustness experiment R2 (PREREGISTRATION_R2.md).

Nothing here writes to disk. A corruption spec is applied to a copy of the
loaded networkx graph with networkx edge operations; the GraphDatabase API,
which auto-persists, is never called. Retrievers are rebuilt on a
CorruptedService whose graph_db.graph is that copy. The vector store is shared
read-only. Calls that could reach the base graph or persist anything raise.

Spec (one per query, a list of operations):
    {"op": "substitute", "relation": "maintained_by", "source": "equipment_TX-440",
     "true_targets": ["procedure_MP-101", "procedure_MP-102"],
     "corrupt_targets": ["procedure_MP-107", "procedure_MP-108"]}
    {"op": "add_spurious", "relation": "maintained_by", "source": "equipment_PD-90",
     "corrupt_targets": ["procedure_MP-119"]}

The reverse relation (maintains, previous_station) changes with the forward one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from graph_db.graph_db import GraphDatabase

CORRUPTION_VERSION = "corrupt-1.0"
INVERSE = {"maintained_by": "maintains", "next_station": "previous_station", "caused_by": None}
CORRUPTIBLE = tuple(INVERSE)
REFUSED_GRAPH_DB = {
    "create_node", "update_node", "delete_node", "add_edge", "create_edge", "create_relationship",
    "delete_edge", "delete_relationship", "save", "persist", "load", "clear",
}
REFUSED_SERVICE = {"hybrid_search", "add_document", "ingest", "delete_document"}


# ============================== Edge helpers ================================


def edges_of(graph: Any, source: str, relation: str) -> List[str]:
    return sorted(target for _, target, data in graph.out_edges(source, data=True)
                  if data.get("type") == relation)


def remove_typed_edge(graph: Any, source: str, target: str, relation: str) -> int:
    removed = 0
    if graph.has_edge(source, target):
        for key, data in list(graph.get_edge_data(source, target).items()):
            if data.get("type") == relation:
                graph.remove_edge(source, target, key)
                removed += 1
    return removed


def add_typed_edge(graph: Any, source: str, target: str, relation: str, tag: str) -> None:
    graph.add_edge(source, target, id=f"r2_{tag}_{source}_{relation}_{target}", type=relation, weight=1.0)


def apply_spec(graph: Any, spec: List[Dict[str, Any]], tag: str = "x") -> List[Dict[str, Any]]:
    """Apply operations to `graph` (a copy). Returns a log of every edge change."""
    log: List[Dict[str, Any]] = []
    for operation in spec:
        relation, source = operation["relation"], operation["source"]
        if relation not in INVERSE:
            raise ValueError(f"relation {relation!r} is not corruptible in R2")
        inverse = INVERSE[relation]
        for node in [source, *operation.get("true_targets", []), *operation["corrupt_targets"]]:
            if node not in graph:
                raise ValueError(f"unknown node {node!r}")
        if operation["op"] == "substitute":
            for target in operation["true_targets"]:
                if remove_typed_edge(graph, source, target, relation) == 0:
                    raise ValueError(f"no {relation} edge {source} -> {target} to substitute")
                log.append({"change": "removed", "source": source, "relation": relation, "target": target})
                if inverse:
                    remove_typed_edge(graph, target, source, inverse)
                    log.append({"change": "removed", "source": target, "relation": inverse, "target": source})
        elif operation["op"] != "add_spurious":
            raise ValueError(f"unknown op {operation['op']!r}")
        for target in operation["corrupt_targets"]:
            add_typed_edge(graph, source, target, relation, tag)
            log.append({"change": "added", "source": source, "relation": relation, "target": target})
            if inverse:
                add_typed_edge(graph, target, source, inverse, tag)
                log.append({"change": "added", "source": target, "relation": inverse, "target": source})
    return log


def edge_set(graph: Any) -> Set[Tuple[str, str, str]]:
    return {(source, str(data.get("type")), target) for source, target, data in graph.edges(data=True)}


def spec_sha256(items: Iterable[Dict[str, Any]]) -> str:
    specs = {str(item["id"]): item.get("corruption") or [] for item in items}
    return hashlib.sha256(json.dumps(specs, sort_keys=True).encode("utf-8")).hexdigest()


# ============================ Corrupted service =============================


class GraphView:
    """Read-only stand-in for GraphDatabase over a corrupted copy."""

    def __init__(self, base: Any, graph: Any):
        self._base = base
        self.graph = graph
        self.db_path = getattr(base, "db_path", None)

    def get_node(self, node_id: str) -> Any:
        return GraphDatabase.get_node(self, node_id)  # reads self.graph only

    def __getattr__(self, name: str) -> Any:
        if name in REFUSED_GRAPH_DB:
            raise AttributeError(f"R2 graph view refuses {name!r}: it could persist or touch the base graph")
        raise AttributeError(f"R2 graph view does not expose {name!r}")


class CorruptedService:
    """Service stand-in: corrupted graph copy, shared read-only vector store."""

    def __init__(self, base: Any, spec: List[Dict[str, Any]], tag: str = "x"):
        self._base = base
        graph = copy.deepcopy(base.graph_db.graph)
        self.change_log = apply_spec(graph, spec, tag)
        self.graph_db = GraphView(base.graph_db, graph)
        self.vector_db = base.vector_db
        self.graph_db_path = getattr(base, "graph_db_path", None)
        self.vector_db_path = getattr(base, "vector_db_path", None)

    def __getattr__(self, name: str) -> Any:
        if name in REFUSED_SERVICE:
            raise AttributeError(f"R2 corrupted service refuses {name!r}: it would read the base graph")
        raise AttributeError(f"R2 corrupted service does not expose {name!r}")


# ========================= Document-truth predicates ========================


def code_of(node_id: str) -> str:
    return node_id.split("_", 1)[1]


def line_orders(texts: Dict[str, str]) -> Dict[str, List[str]]:
    """Station order per line, read from line text: 'runs from X to Y, calling at A, B, C.'"""
    orders: Dict[str, List[str]] = {}
    for node_id, text in texts.items():
        if not node_id.startswith("line_"):
            continue
        match = re.match(r"The \w+ Line runs from (.+?) to (.+?), calling at (.+?)\. ", text)
        if not match:
            continue
        start, end, middle = match.groups()
        orders[node_id] = [start, *[name.strip() for name in middle.split(",")], end]
    return orders


def station_id(name: str) -> str:
    return "station_" + name.lower().replace(" ", "_")


def document_supports(relation: str, source: str, target: str, texts: Dict[str, str],
                      orders: Optional[Dict[str, List[str]]] = None) -> bool:
    """Does the source text state this relation?"""
    if relation == "caused_by":
        return re.search(rf"(?<![\w-]){re.escape(code_of(target))}(?![\w-])", texts.get(source, "")) is not None
    if relation == "maintained_by":
        return f"Applies to the {code_of(source)} " in texts.get(target, "")
    if relation == "next_station":
        for order in (orders or line_orders(texts)).values():
            ids = [station_id(name) for name in order]
            for first, second in zip(ids, ids[1:]):
                if (first, second) == (source, target):
                    return True
        return False
    raise ValueError(f"no document predicate for {relation!r}")
