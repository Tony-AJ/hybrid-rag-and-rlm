"""Deterministic subject-relation-object verifier (R2', relver-1.0).

Preregistered in PREREGISTRATION_R2_PRIME.md. No model calls. Everything
corpus-specific lives in relation_schema.json: entity types and how they are
mentioned in text, each relation's check kind and witness endpoint, and the
data-format adapters for ordered container texts. This module contains no
relation phrase.

Labels per claim:
  supported      the witness text states the relation (mentions the counterpart,
                 or lists the pair adjacently in the stated direction)
  contradicted   the witness text is complete for that slot (schema: exhaustive)
                 and names a different entity of the counterpart type instead
  unverified     the witness text is unavailable, names no entity of that type,
                 or is not exhaustive
  graph_only     the relation is not stated by any document (schema); permitted

Deviations from PREREGISTRATION_R2_PRIME.md, made before any sweep, model call or freeze
(found in design review, not from data):
  D1  completeness: besides supported hops of the same type pair, a contradicted hop's
      witness on the path's fixed side contributes the entities it names, and a sequence
      container contributes the true neighbour, as missing slot fillers. This makes drops
      stricter (more abstentions), never looser.
  D2  entity status is decided on the shortest anchor-to-entity paths only, so a longer
      chain through unrelated shared entities cannot override a direct contradicted link.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

VERIFIER_VERSION = "relver-1.0"
ABSTAIN = "INSUFFICIENT EVIDENCE"
MAX_PATH_HOPS = 3
_EVIDENCE_LINE = re.compile(r"^\[\d+\] \(([^)]+)\) (.*)$")
_DASHES = re.compile(r"[‐-―−]")
_SPACES = re.compile(r"[  -   　]")

Claim = Tuple[str, str, str]


def normalise(text: str) -> str:
    return _SPACES.sub(" ", _DASHES.sub("-", text)).replace("_", " ")


def load_schema(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ================================ Registry ==================================


class Registry:
    """Entity types and text labels, derived from node ids and texts by schema rules."""

    def __init__(self, schema: Dict[str, Any], texts: Dict[str, str]):
        self.schema = schema
        self.types: Dict[str, str] = {}
        self.labels: Dict[str, str] = {}
        for node_id, text in texts.items():
            etype = next((name for name, spec in schema["entity_types"].items()
                          if node_id.startswith(spec["id_prefix"])), None)
            if etype is None:
                continue
            self.types[node_id] = etype
            label = self._label(node_id, text, schema["entity_types"][etype]["label"])
            if label:
                self.labels[node_id] = label
        # Case-insensitive: answers often cite node ids (station_kestrel_green), which
        # normalise to "station kestrel green".
        self._patterns = {node_id: re.compile(rf"(?<![\w-]){re.escape(normalise(label))}(?![\w-])", re.IGNORECASE)
                          for node_id, label in self.labels.items()}

    @staticmethod
    def _label(node_id: str, text: str, rule: Dict[str, Any]) -> Optional[str]:
        kind = rule["kind"]
        if kind == "id_suffix":
            return node_id.split("_", 1)[1]
        if kind == "id_pattern":
            match = re.match(rule["pattern"], node_id)
            return rule["format"].format(*match.groups()) if match else None
        if kind == "text_pattern":
            match = re.match(rule["pattern"], text or "")
            return match.group(1) if match else None
        return None

    def type_of(self, node_id: str) -> Optional[str]:
        return self.types.get(node_id)

    def mentions(self, text: str, etype: Optional[str] = None) -> Set[str]:
        folded = normalise(text or "")
        return {node_id for node_id, pattern in self._patterns.items()
                if (etype is None or self.types[node_id] == etype) and pattern.search(folded)}


# ================================ Verifier ==================================


class Verifier:
    def __init__(self, schema: Dict[str, Any], registry: Registry,
                 fetch_text: Callable[[str], Optional[str]],
                 relations: Callable[[str], Dict[str, List[str]]]):
        self.schema = schema
        self.registry = registry
        self._fetch = fetch_text
        self._relations = relations
        self._text_cache: Dict[str, Optional[str]] = {}
        self._cache: Dict[Claim, Dict[str, Any]] = {}

    def text(self, node_id: str) -> Optional[str]:
        if node_id not in self._text_cache:
            text = self._fetch(node_id)
            self._text_cache[node_id] = text if text and text.strip() else None
        return self._text_cache[node_id]

    def verify(self, subject: str, relation: str, obj: str) -> Dict[str, Any]:
        key = (subject, relation, obj)
        if key not in self._cache:
            self._cache[key] = self._verify(subject, relation, obj)
        return self._cache[key]

    def _verify(self, subject: str, relation: str, obj: str) -> Dict[str, Any]:
        spec = self.schema["relations"].get(relation)
        if spec is None:
            return {"status": "unverified", "reason": "relation_not_in_schema"}
        if self.registry.type_of(subject) != spec["subject"] or self.registry.type_of(obj) != spec["object"]:
            return {"status": "unverified", "reason": "endpoint_type_mismatch"}
        check = spec["check"]
        if check == "not_text_verifiable":
            return {"status": "graph_only", "reason": "relation_not_stated_by_documents"}
        if check == "endpoint_mention":
            witness, counterpart = (subject, obj) if spec["witness"] == "subject" else (obj, subject)
            text = self.text(witness)
            if text is None:
                return {"status": "unverified", "reason": "witness_unavailable", "witness": witness}
            named = self.registry.mentions(text, self.registry.type_of(counterpart)) - {witness}
            if counterpart in named:
                return {"status": "supported", "reason": "witness_mentions_counterpart", "witness": witness}
            if spec.get("exhaustive") and named:
                return {"status": "contradicted", "reason": "exhaustive_witness_names_other",
                        "witness": witness, "witness_names": sorted(named)}
            return {"status": "unverified", "reason": "witness_silent_or_not_exhaustive", "witness": witness}
        if check == "sequence":
            return self._verify_sequence(subject, obj, spec)
        return {"status": "unverified", "reason": f"unknown_check_{check}"}

    def _verify_sequence(self, subject: str, obj: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        extractor = self.schema["sequence_extractors"][spec["extractor"]]
        subject_label = self.registry.labels.get(subject)
        containers = self._relations(subject).get(spec["container_relation"], [])
        for container in containers:
            text = self.text(container)
            if text is None:
                continue
            order = extract_order(text, extractor)
            if subject_label not in order:
                continue
            index = order.index(subject_label)
            step = 1 if spec["direction"] == "forward" else -1
            neighbour_index = index + step
            neighbour = order[neighbour_index] if 0 <= neighbour_index < len(order) else None
            if neighbour == self.registry.labels.get(obj):
                return {"status": "supported", "reason": "adjacent_in_container_order", "witness": container}
            return {"status": "contradicted", "reason": "container_order_has_other_neighbour",
                    "witness": container, "neighbour": neighbour}
        return {"status": "unverified", "reason": "no_container_text_lists_subject", "containers": containers}


def extract_order(text: str, extractor: Dict[str, Any]) -> List[str]:
    match = re.search(extractor["pattern"], text)
    if not match:
        return []
    order: List[str] = []
    for group in extractor["order"]:
        value = match.group(group)
        order.extend(part.strip() for part in value.split(extractor["separator"])) if group == "middle" else order.append(value.strip())
    return order


# ============================ Received evidence =============================


def parse_prompts(prompts: Iterable[str]) -> Dict[str, Any]:
    """Evidence chunks and Links edges the model actually received."""
    nodes: List[str] = []
    texts: Dict[str, str] = {}
    links: Set[Claim] = set()
    for prompt in prompts:
        for line in prompt.splitlines():
            match = _EVIDENCE_LINE.match(line.strip())
            if not match:
                continue
            node_id, rest = match.groups()
            body, _, link_part = rest.partition(" Links: ")
            if node_id not in texts:
                nodes.append(node_id)
            texts[node_id] = body
            for segment in filter(None, (part.strip() for part in link_part.split(";"))):
                relation, _, targets = segment.partition(" -> ")
                for target in filter(None, (t.strip() for t in targets.split(","))):
                    links.add((node_id, relation.strip(), target))
    return {"nodes": nodes, "texts": texts, "links": links}


# ============================ Claims and paths ==============================


@dataclass
class Hop:
    claim: Claim
    source: str  # "link" or "text"
    verdict: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.verdict["status"]


def collect_hops(verifier: Verifier, received: Dict[str, Any], pool: Set[str]) -> Dict[Claim, Hop]:
    hops: Dict[Claim, Hop] = {}
    for claim in received["links"]:
        hops[claim] = Hop(claim, "link", verifier.verify(*claim))
    relations = verifier.schema["relations"]
    registry = verifier.registry
    for subject in pool:
        for obj in pool:
            if subject == obj:
                continue
            for relation, spec in relations.items():
                if spec["check"] == "not_text_verifiable":
                    continue
                if registry.type_of(subject) != spec["subject"] or registry.type_of(obj) != spec["object"]:
                    continue
                claim = (subject, relation, obj)
                if claim in hops:
                    continue
                verdict = verifier.verify(*claim)
                if verdict["status"] == "supported":
                    hops[claim] = Hop(claim, "text", verdict)
    return hops


def enumerate_paths(hops: Dict[Claim, Hop], start: str, goal: str, max_hops: int = MAX_PATH_HOPS) -> List[List[Hop]]:
    adjacency: Dict[str, List[Tuple[str, Hop]]] = {}
    for hop in hops.values():
        subject, _, obj = hop.claim
        adjacency.setdefault(subject, []).append((obj, hop))
        adjacency.setdefault(obj, []).append((subject, hop))
    paths: List[List[Hop]] = []

    def walk(node: str, visited: Set[str], path: List[Hop]) -> None:
        if node == goal and path:
            paths.append(list(path))
            return
        if len(path) == max_hops:
            return
        for neighbour, hop in adjacency.get(node, []):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            path.append(hop)
            walk(neighbour, visited, path)
            path.pop()
            visited.discard(neighbour)

    walk(start, {start}, [])
    return paths


def path_status(path: List[Hop]) -> str:
    statuses = [hop.status for hop in path]
    if all(status in ("supported", "graph_only") for status in statuses):
        return "graph_only" if "graph_only" in statuses else "supported"
    if "contradicted" in statuses:
        return "contradicted"
    return "unverified"


def path_nodes(path: List[Hop], start: str) -> List[str]:
    nodes = [start]
    for hop in path:
        subject, _, obj = hop.claim
        nodes.append(obj if nodes[-1] == subject else subject)
    return nodes


# ================================ Decision ==================================


def verify_answer(answer: str, question: str, prompts: Iterable[str], verifier: Verifier) -> Dict[str, Any]:
    prompts = list(prompts)
    registry = verifier.registry
    if ABSTAIN in normalise(answer).upper():
        return {"verifier_version": VERIFIER_VERSION, "decision": "keep", "reason": "answer_abstains",
                "final_answer": answer, "entities": {}, "claims": []}
    anchors = registry.mentions(question)
    answer_entities = registry.mentions(answer) - anchors
    if not answer_entities:
        return {"verifier_version": VERIFIER_VERSION, "decision": "keep", "reason": "no_answer_entities",
                "final_answer": answer, "anchors": sorted(anchors), "entities": {}, "claims": []}

    received = parse_prompts(prompts)
    pool = set(anchors) | set(received["nodes"]) | set(answer_entities)
    for subject, _, obj in received["links"]:
        pool |= {subject, obj}
    for text in received["texts"].values():
        pool |= registry.mentions(text)
    pool = {node for node in pool if registry.type_of(node)}
    hops = collect_hops(verifier, received, pool)

    entities: Dict[str, Any] = {}
    all_paths: Dict[str, List[Tuple[str, List[Hop]]]] = {}
    for entity in sorted(answer_entities):
        found = [(anchor, path) for anchor in sorted(anchors) for path in enumerate_paths(hops, anchor, entity)]
        # Only the most direct connection counts: a longer chain through unrelated shared
        # entities (two incidents on one line) must not override a direct contradicted link.
        if found:
            shortest = min(len(path) for _, path in found)
            found = [(anchor, path) for anchor, path in found if len(path) == shortest]
        all_paths[entity] = found
        statuses = [path_status(path) for _, path in found]
        if "supported" in statuses:
            status = "supported"
        elif "graph_only" in statuses:
            status = "graph_only"
        elif "contradicted" in statuses:
            status = "contradicted"
        else:
            status = "unverified"
        chosen = next(((a, p) for (a, p), s in zip(found, statuses) if s == status), None)
        entities[entity] = {
            "status": status, "label": registry.labels.get(entity), "type": registry.type_of(entity),
            "paths_found": len(found),
            "anchor": chosen[0] if chosen else None,
            "path_nodes": path_nodes(chosen[1], chosen[0]) if chosen else [],
            "path": [{"claim": list(hop.claim), "source": hop.source, **hop.verdict} for hop in chosen[1]] if chosen else [],
        }

    # Terminal entities: not an interior node on another answer entity's path.
    interior: Set[str] = set()
    for entity, found in all_paths.items():
        for anchor, path in found:
            interior |= set(path_nodes(path, anchor)[1:-1]) - {entity}
    terminal = {entity for entity in answer_entities if entity not in interior}

    statuses = {entity: info["status"] for entity, info in entities.items()}
    contradicted = sorted(e for e, s in statuses.items() if s == "contradicted")
    unverified = sorted(e for e, s in statuses.items() if s == "unverified")
    if unverified:
        decision, reason, final = "abstain", "unverified_entities", ABSTAIN
    elif not contradicted:
        decision, reason, final = "keep", "all_supported_or_graph_only", answer
    else:
        remaining = sorted(e for e in terminal if statuses[e] in ("supported", "graph_only"))
        gaps = completeness_gaps(hops, entities, contradicted, answer_entities, verifier)
        if remaining and not gaps:
            labels = [registry.labels.get(e, e) for e in remaining]
            decision, reason, final = "drop", "contradicted_dropped_rest_complete", "Verified: " + ", ".join(labels) + "."
        else:
            decision = "abstain"
            reason = "nothing_supported_remains" if not remaining else "remaining_answer_incomplete"
            final = ABSTAIN
    claims = sorted({tuple(step["claim"]) for info in entities.values() for step in info["path"]})
    claim_records = [{"claim": list(claim), "source": hops[claim].source, **hops[claim].verdict} for claim in claims]
    return {
        "verifier_version": VERIFIER_VERSION, "decision": decision, "reason": reason, "final_answer": final,
        "anchors": sorted(anchors), "answer_entities": sorted(answer_entities), "terminal": sorted(terminal),
        "entities": entities, "contradicted": contradicted, "unverified": unverified,
        "completeness_gaps": completeness_gaps(hops, entities, contradicted, answer_entities, verifier) if contradicted else [],
        "claims": claim_records,
    }


def completeness_gaps(hops: Dict[Claim, Hop], entities: Dict[str, Any], contradicted: List[str],
                      answer_entities: Set[str], verifier: Verifier) -> List[str]:
    """Entities the evidence supports for a contradicted claim's slot that the answer omits.

    The slot is oriented the way the path walks from the anchor: the node before the
    contradicted hop is fixed, and any other node of the next node's type connected
    to it by a supported (or graph_only) hop of the same type pair fills the slot.
    """
    registry, relations = verifier.registry, verifier.schema["relations"]
    gaps: Set[str] = set()
    for entity in contradicted:
        info = entities[entity]
        nodes = info["path_nodes"]
        for index, step in enumerate(info["path"]):
            if step["status"] != "contradicted":
                continue
            fixed, wrong = nodes[index], nodes[index + 1]
            # The witness text may itself name the true slot filler: an exhaustive witness on
            # the fixed side names it (witness_names), a sequence container names the neighbour.
            if step.get("witness") == fixed:
                gaps |= {node for node in step.get("witness_names", []) if node not in answer_entities}
            if step.get("neighbour"):
                gaps |= {node for node, label in registry.labels.items()
                         if label == step["neighbour"] and registry.type_of(node) == registry.type_of(wrong)
                         and node not in answer_entities}
            pair = {registry.type_of(fixed), registry.type_of(wrong)}
            family = {name for name, spec in relations.items() if {spec["subject"], spec["object"]} == pair}
            for (s, r, o), hop in hops.items():
                if hop.status not in ("supported", "graph_only") or r not in family or fixed not in (s, o):
                    continue
                other = o if s == fixed else s
                if other != wrong and registry.type_of(other) == registry.type_of(wrong) and other not in answer_entities:
                    gaps.add(other)
    return sorted(gaps)
