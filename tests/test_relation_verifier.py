"""Unit tests for rlm.relation_verifier (relver-1.0) and rlm.robustness_scoring_r2p. Toy texts; no stores."""

from pathlib import Path

import pytest

from rlm import relation_verifier as rv
from rlm import robustness_scoring_r2p as r2ps

SCHEMA = rv.load_schema(Path(__file__).resolve().parents[1] / "benchmark_corpora/meridian/robustness/relation_schema.json")

TEXTS = {
    "equipment_TX-440": "The TX-440 traction inverter is fitted across the fleet.",
    "equipment_PD-90": "The PD-90 platform screen door controller is fitted across the fleet.",
    "procedure_MP-101": "Maintenance procedure MP-101: Phase check. Applies to the TX-440 traction inverter.",
    "procedure_MP-102": "Maintenance procedure MP-102: Gate driver. Applies to the TX-440 traction inverter. This procedure supersedes MP-101.",
    "procedure_MP-107": "Maintenance procedure MP-107: Door survey. Applies to the PD-90 platform screen door controller.",
    "incident_2026_01": "Service incident 2026/01 on 2026-01-04: trains were held at Harbour Row on the Azure Line after a TX-440 traction inverter tripped.",
    "station_harbour_row": "Harbour Row is a terminus on the Azure Line with 3 platforms.",
    "station_kestrel_green": "Kestrel Green is a through station on the Azure Line.",
    "station_alder_vale": "Alder Vale is a terminus on the Azure Line.",
    "station_glasshouse": "Glasshouse is a terminus on the Slate Line.",
    "line_azure": "The Azure Line runs from Harbour Row to Alder Vale, calling at Kestrel Green. It is operated with sets.",
    "line_slate": "The Slate Line runs from Glasshouse to Ivory Dock, calling at Hartsmere. It is operated with sets.",
    "policy_PL-A": "Authorisation policy PL-A, Traction Maintenance Authorisation, governs propulsion work.",
}
SERVED_BY = {"station_harbour_row": ["line_azure"], "station_kestrel_green": ["line_azure"],
             "station_alder_vale": ["line_azure"], "station_glasshouse": ["line_slate"]}


def make(masked=()):
    registry = rv.Registry(SCHEMA, TEXTS)
    return rv.Verifier(SCHEMA, registry, lambda n: None if n in masked else TEXTS.get(n),
                       lambda n: {"served_by": SERVED_BY.get(n, [])})


def chunk_line(index, node, links=None):
    line = f"[{index}] ({node}) {TEXTS[node]}"
    if links:
        line += " Links: " + "; ".join(f"{rel} -> {', '.join(targets)}" for rel, targets in links.items())
    return line


def prompt(*lines):
    return "Answer using only the evidence.\nQuestion: q\nEvidence:\n" + "\n".join(lines) + "\nAnswer:"


# ------------------------------ labels -------------------------------------

def test_registry_labels():
    registry = rv.Registry(SCHEMA, TEXTS)
    assert registry.labels["incident_2026_01"] == "2026/01"
    assert registry.labels["line_azure"] == "Azure Line"
    assert registry.labels["station_kestrel_green"] == "Kestrel Green"
    assert registry.mentions("see station_kestrel_green", "station") == {"station_kestrel_green"}


def test_endpoint_supported_and_contradicted():
    v = make()
    assert v.verify("equipment_TX-440", "maintained_by", "procedure_MP-101")["status"] == "supported"
    verdict = v.verify("equipment_TX-440", "maintained_by", "procedure_MP-107")
    assert verdict["status"] == "contradicted" and verdict["witness_names"] == ["equipment_PD-90"]
    assert v.verify("procedure_MP-107", "maintains", "equipment_TX-440")["status"] == "contradicted"
    assert v.verify("incident_2026_01", "caused_by", "equipment_PD-90")["status"] == "contradicted"
    assert v.verify("incident_2026_01", "caused_by", "equipment_TX-440")["status"] == "supported"


def test_unavailable_witness_is_unverified_not_contradicted():
    v = make(masked={"procedure_MP-107"})
    assert v.verify("equipment_TX-440", "maintained_by", "procedure_MP-107")["status"] == "unverified"


def test_non_exhaustive_witness_never_contradicts():
    v = make()
    assert v.verify("procedure_MP-102", "supersedes", "procedure_MP-101")["status"] == "supported"
    assert v.verify("procedure_MP-102", "supersedes", "procedure_MP-107")["status"] == "unverified"


def test_sequence_is_directional():
    v = make()
    assert v.verify("station_harbour_row", "next_station", "station_kestrel_green")["status"] == "supported"
    assert v.verify("station_kestrel_green", "previous_station", "station_harbour_row")["status"] == "supported"
    assert v.verify("station_kestrel_green", "next_station", "station_harbour_row")["status"] == "contradicted"
    assert v.verify("station_kestrel_green", "next_station", "station_glasshouse")["status"] == "contradicted"
    assert v.verify("station_alder_vale", "next_station", "station_kestrel_green")["status"] == "contradicted"


def test_sequence_without_container_text_is_unverified():
    v = make(masked={"line_azure"})
    assert v.verify("station_harbour_row", "next_station", "station_kestrel_green")["status"] == "unverified"


def test_graph_only_and_type_mismatch():
    v = make()
    assert v.verify("procedure_MP-101", "authorised_by", "policy_PL-A")["status"] == "graph_only"
    assert v.verify("procedure_MP-101", "maintained_by", "equipment_TX-440")["status"] == "unverified"


def test_parse_prompts_links():
    received = rv.parse_prompts([prompt(chunk_line(1, "equipment_TX-440", {"maintained_by": ["procedure_MP-101", "procedure_MP-107"]}))])
    assert received["nodes"] == ["equipment_TX-440"]
    assert ("equipment_TX-440", "maintained_by", "procedure_MP-107") in received["links"]
    assert "Links" not in received["texts"]["equipment_TX-440"]


# ------------------------------ decisions ----------------------------------

EQ_LINKS = chunk_line(1, "equipment_TX-440", {"maintained_by": ["procedure_MP-101", "procedure_MP-107"]})
Q_EQ = "Which maintenance procedures apply to the TX-440?"


def test_keep_when_supported():
    result = rv.verify_answer("MP-101.", Q_EQ, [prompt(EQ_LINKS)], make())
    assert result["decision"] == "keep"
    assert result["entities"]["procedure_MP-101"]["status"] == "supported"


def test_drop_spurious_entity_when_rest_complete():
    result = rv.verify_answer("MP-101 and MP-107.", Q_EQ, [prompt(EQ_LINKS)], make())
    assert result["decision"] == "drop"
    assert result["final_answer"] == "Verified: MP-101."
    assert result["entities"]["procedure_MP-107"]["status"] == "contradicted"


def test_abstain_when_remaining_answer_incomplete():
    # MP-102 is a supported link target for the same slot but missing from the answer.
    links = chunk_line(1, "equipment_TX-440", {"maintained_by": ["procedure_MP-101", "procedure_MP-102", "procedure_MP-107"]})
    result = rv.verify_answer("MP-101 and MP-107.", Q_EQ, [prompt(links)], make())
    assert result["decision"] == "abstain" and result["reason"] == "remaining_answer_incomplete"
    assert result["completeness_gaps"] == ["procedure_MP-102"]


def test_abstain_when_everything_contradicted():
    links = chunk_line(1, "equipment_TX-440", {"maintained_by": ["procedure_MP-107"]})
    result = rv.verify_answer("MP-107.", Q_EQ, [prompt(links)], make())
    assert result["decision"] == "abstain" and result["final_answer"] == rv.ABSTAIN


def test_abstain_when_unverified():
    result = rv.verify_answer("MP-107.", Q_EQ, [prompt(EQ_LINKS)], make(masked={"procedure_MP-107"}))
    assert result["decision"] == "abstain" and result["reason"] == "unverified_entities"


def test_every_edge_of_two_hop_path_verified():
    # Corrupted first hop: incident caused_by PD-90 (text says TX-440); second hop PD-90 -> MP-107 is true.
    lines = [chunk_line(1, "incident_2026_01", {"caused_by": ["equipment_PD-90"]}),
             chunk_line(2, "equipment_PD-90", {"maintained_by": ["procedure_MP-107"]})]
    result = rv.verify_answer("MP-107 covers it.", "Which procedures cover the equipment that failed in service incident 2026/01?",
                              [prompt(*lines)], make())
    entity = result["entities"]["procedure_MP-107"]
    assert entity["status"] == "contradicted"
    statuses = [step["status"] for step in entity["path"]]
    assert statuses == ["contradicted", "supported"]
    assert result["decision"] == "abstain"


def test_text_derived_path_supports_answer_without_links():
    lines = [chunk_line(1, "incident_2026_01"), chunk_line(2, "procedure_MP-101")]
    result = rv.verify_answer("MP-101.", "Which procedures cover the equipment that failed in service incident 2026/01?",
                              [prompt(*lines)], make())
    assert result["entities"]["procedure_MP-101"]["status"] == "supported"
    assert {step["source"] for step in result["entities"]["procedure_MP-101"]["path"]} == {"text"}


def test_graph_only_answer_kept():
    lines = [chunk_line(1, "procedure_MP-101", {"authorised_by": ["policy_PL-A"]})]
    result = rv.verify_answer("PL-A.", "Which policy covers maintenance procedure MP-101?", [prompt(*lines)], make())
    assert result["entities"]["policy_PL-A"]["status"] == "graph_only"
    assert result["decision"] == "keep"


def test_adjacency_answer_with_node_ids():
    lines = [chunk_line(1, "station_kestrel_green", {"next_station": ["station_glasshouse"], "previous_station": ["station_harbour_row"]})]
    result = rv.verify_answer("station_harbour_row and station_glasshouse.",
                              "Which station comes immediately before Kestrel Green and which after it?", [prompt(*lines)], make())
    assert result["entities"]["station_harbour_row"]["status"] == "supported"
    assert result["entities"]["station_glasshouse"]["status"] == "contradicted"


def test_adjacency_drop_blocked_when_container_names_true_neighbour():
    # Harbour Row is supported; Glasshouse contradicted; the line text gives the true next
    # station (Alder Vale), so dropping Glasshouse would leave an incomplete answer.
    lines = [chunk_line(1, "station_kestrel_green", {"next_station": ["station_glasshouse"], "previous_station": ["station_harbour_row"]})]
    result = rv.verify_answer("station_harbour_row and station_glasshouse.",
                              "Which station comes immediately before Kestrel Green and which after it?", [prompt(*lines)], make())
    assert result["decision"] == "abstain"
    assert "station_alder_vale" in result["completeness_gaps"]


def test_caused_by_witness_names_true_filler():
    lines = [chunk_line(1, "incident_2026_01", {"caused_by": ["equipment_PD-90"]})]
    result = rv.verify_answer("PD-90.", "Which component failed in service incident 2026/01?", [prompt(*lines)], make())
    assert result["entities"]["equipment_PD-90"]["status"] == "contradicted"
    assert result["decision"] == "abstain"


def test_shortest_path_decides():
    # A direct contradicted link is not overridden by a longer supported chain.
    hops = {
        ("a", "r", "x"): rv.Hop(("a", "r", "x"), "link", {"status": "contradicted"}),
        ("a", "r", "b"): rv.Hop(("a", "r", "b"), "text", {"status": "supported"}),
        ("b", "r", "x"): rv.Hop(("b", "r", "x"), "text", {"status": "supported"}),
    }
    paths = rv.enumerate_paths(hops, "a", "x")
    assert sorted(len(p) for p in paths) == [1, 2]


def test_abstaining_answer_is_kept():
    assert rv.verify_answer("INSUFFICIENT EVIDENCE", Q_EQ, [prompt(EQ_LINKS)], make())["decision"] == "keep"


def test_deterministic():
    a = rv.verify_answer("MP-101 and MP-107.", Q_EQ, [prompt(EQ_LINKS)], make())
    b = rv.verify_answer("MP-101 and MP-107.", Q_EQ, [prompt(EQ_LINKS)], make())
    assert a == b


# ------------------------------- scoring -----------------------------------

@pytest.mark.parametrize("true,status,category", [
    (True, "supported", "correct_support"), (True, "contradicted", "false_contradiction"),
    (True, "unverified", "missed_support"), (True, "graph_only", "permitted"),
    (False, "supported", "false_support"), (False, "contradicted", "contradiction_detected"),
    (False, "unverified", "safe_miss"), (False, "graph_only", "false_allowance"),
])
def test_claim_categories(true, status, category):
    assert r2ps.claim_category(true, status) == category


def test_relation_summary_rates():
    clean = {("equipment_TX-440", "maintained_by", "procedure_MP-101")}
    claims = r2ps.score_claims([
        {"claim": ["equipment_TX-440", "maintained_by", "procedure_MP-101"], "status": "supported"},
        {"claim": ["equipment_TX-440", "maintained_by", "procedure_MP-107"], "status": "contradicted"},
        {"claim": ["equipment_TX-440", "maintained_by", "procedure_MP-108"], "status": "unverified"},
    ], clean)
    summary = r2ps.relation_summary(claims)["overall"]
    assert summary["contradiction_detection"] == "1/2 (50.0%)"
    assert summary["false_contradiction"] == "0/1 (0.0%)"
    assert summary["verification_accuracy"] == "2/3 (66.7%)"
