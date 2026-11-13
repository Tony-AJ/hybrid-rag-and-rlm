"""Unit tests for rlm.robustness_scoring_r2. Pure functions, no stores, no model."""

import pytest

from rlm import robustness_scoring_r2 as r2s

UNIVERSE = ["Harbour Row", "Alder Vale", "Kestrel Green", "Brickworks", "Glasshouse"]

EQUIPMENT = ("[1] (equipment_TX-440) The TX-440 traction inverter. "
             "Links: maintained_by -> procedure_MP-107, procedure_MP-108")
TRUE_PROC = "[2] (procedure_MP-101) Maintenance procedure MP-101: x. Applies to the TX-440 traction inverter."
TRUE_PROC2 = "[3] (procedure_MP-102) Maintenance procedure MP-102: y. Applies to the TX-440 traction inverter."
WRONG_PROC = "[4] (procedure_MP-107) Maintenance procedure MP-107: z. Applies to the PD-90 pantograph."
LINE = ("[1] (line_azure) The Azure Line runs from Harbour Row to Dunmore Park, calling at Kestrel Green, "
        "Alder Vale, Brickworks, Copper Lane.")
STATION = "[2] (station_alder_vale) Alder Vale is a through station. Links: next_station -> station_glasshouse"

R2B = {"id": 2007, "case": "R2b", "answerable": True,
       "query": "Which maintenance procedures cover the fault that stopped service at Harbour Row?",
       "corruption": [{"op": "substitute", "relation": "maintained_by", "source": "equipment_TX-440",
                       "true_targets": ["procedure_MP-101", "procedure_MP-102"],
                       "corrupt_targets": ["procedure_MP-107", "procedure_MP-108"]}],
       "document_truth": {"entity_type": "procedure_code", "entities": ["MP-101", "MP-102"]},
       "graph_claim": {"entity_type": "procedure_code", "entities": ["MP-107", "MP-108"]}}
R2C = {"id": 2015, "case": "R2c", "answerable": True,
       "query": "Which stations sit immediately either side of Alder Vale along its own line?",
       "corruption": [{"op": "substitute", "relation": "next_station", "source": "station_alder_vale",
                       "true_targets": ["station_brickworks"], "corrupt_targets": ["station_glasshouse"]}],
       "document_truth": {"entity_type": "station_name", "entities": ["Brickworks", "Kestrel Green"]},
       "graph_claim": {"entity_type": "station_name", "entities": ["Glasshouse"]}}
R2E = {"id": 2027, "case": "R2e", "answerable": False,
       "query": "Which maintenance procedures cover the fault that stopped service at Mallow Street?",
       "corruption": [{"op": "substitute", "relation": "maintained_by", "source": "equipment_TX-440",
                       "true_targets": ["procedure_MP-101", "procedure_MP-102"], "corrupt_targets": ["procedure_MP-107"]}],
       "document_truth": {"entity_type": "procedure_code", "entities": []},
       "graph_claim": {"entity_type": "procedure_code", "entities": ["MP-107"]}}


def prompt(*lines):
    return "Answer using only the evidence.\nQuestion: q\nEvidence:\n" + "\n".join(lines) + "\nAnswer:"


def score(answer, item, *lines):
    return r2s.score_r2(answer, item, [prompt(*lines)], UNIVERSE)


def test_correct_document_supported():
    result = score("MP-101 and MP-102 cover the TX-440 fault.", R2B, EQUIPMENT, TRUE_PROC, TRUE_PROC2)
    assert result["outcome"] == "correct"
    assert result["document_supported"] and result["identifier_supported"]
    assert not result["graph_supported"]


def test_graph_misled_link_only():
    result = score("MP-107 and MP-108.", R2B, EQUIPMENT)
    assert result["outcome"] == "graph_misled"
    assert result["graph_supported"] and result["corrupt_visible_in_links"]
    assert not result["wrong_target_text_received"]


def test_graph_misled_with_wrong_text_read():
    result = score("MP-107.", R2B, EQUIPMENT, WRONG_PROC)
    assert result["outcome"] == "graph_misled"
    assert result["wrong_target_text_received"]
    assert result["identifier_supported"]  # identifier check passes: the R2 blind spot


def test_mixed():
    result = score("MP-101, MP-102 and MP-107.", R2B, EQUIPMENT, TRUE_PROC, TRUE_PROC2)
    assert result["outcome"] == "mixed"


def test_over_abstention():
    result = score("INSUFFICIENT EVIDENCE", R2B, EQUIPMENT)
    assert result["outcome"] == "over_abstention"


def test_correct_abstention_unanswerable():
    result = score("INSUFFICIENT EVIDENCE", R2E, EQUIPMENT)
    assert result["outcome"] == "correct_abstention"


def test_unanswerable_naming_corrupt_target_is_graph_misled():
    result = score("MP-107 covers it.", R2E, EQUIPMENT, WRONG_PROC)
    assert result["outcome"] == "graph_misled"


def test_unsupported_answer():
    result = score("MP-101 covers it.", R2E, EQUIPMENT)
    assert result["outcome"] == "unsupported_answer"


def test_other_wrong():
    result = score("MP-101 only.", R2B, EQUIPMENT, TRUE_PROC, TRUE_PROC2)
    assert result["outcome"] == "other_wrong"


def test_adjacency_question_station_not_counted():
    result = score("Kestrel Green and Brickworks sit either side of Alder Vale.", R2C, LINE, STATION)
    assert result["outcome"] == "correct"
    assert result["named"] == ["Brickworks", "Kestrel Green"]


def test_adjacency_graph_misled():
    result = score("Kestrel Green and Glasshouse.", R2C, LINE, STATION)
    assert result["outcome"] == "graph_misled"
    assert result["graph_supported"]


@pytest.mark.parametrize("outcome", r2s.OUTCOMES)
def test_every_outcome_reachable(outcome):
    examples = {
        "correct": ("MP-101 and MP-102.", R2B, (EQUIPMENT, TRUE_PROC, TRUE_PROC2)),
        "correct_abstention": ("INSUFFICIENT EVIDENCE", R2E, (EQUIPMENT,)),
        "mixed": ("MP-101, MP-102, MP-108.", R2B, (EQUIPMENT, TRUE_PROC, TRUE_PROC2)),
        "graph_misled": ("MP-108.", R2B, (EQUIPMENT,)),
        "over_abstention": ("INSUFFICIENT EVIDENCE", R2B, (EQUIPMENT,)),
        "unsupported_answer": ("MP-101.", R2E, (EQUIPMENT,)),
        "other_wrong": ("MP-102.", R2B, (EQUIPMENT, TRUE_PROC, TRUE_PROC2)),
    }
    answer, item, lines = examples[outcome]
    assert score(answer, item, *lines)["outcome"] == outcome


OTHER_INCIDENT = "[5] (incident_2026_06) Trains were held at Dunmore Park after an AX-3 axle counter failed."


@pytest.mark.parametrize("answer,lines,source", [
    ("MP-107.", (EQUIPMENT,), "links_only"),
    ("MP-107.", (EQUIPMENT, WRONG_PROC), "links_and_text"),
    ("MP-109.", (EQUIPMENT, "[6] (procedure_MP-109) Maintenance procedure MP-109: q. Applies to the SG-12."), "text_only"),
    ("MP-150.", (EQUIPMENT,), "neither"),
    ("MP-101 only.", (EQUIPMENT, TRUE_PROC, TRUE_PROC2), "no_wrong_entity"),
])
def test_wrong_source(answer, lines, source):
    result = score(answer, R2B, *lines)
    assert result["is_wrong"]
    assert result["wrong_source"] == source


def test_right_answers_have_no_wrong_source():
    assert score("MP-101 and MP-102.", R2B, EQUIPMENT, TRUE_PROC, TRUE_PROC2)["wrong_source"] is None
    assert score("INSUFFICIENT EVIDENCE", R2B, EQUIPMENT)["wrong_source"] is None  # over-abstention is not a wrong entity


def test_corrupt_code_in_unrelated_text_is_text_only_not_links():
    item = {**R2B, "graph_claim": {"entity_type": "equipment_code", "entities": ["AX-3"]},
            "document_truth": {"entity_type": "equipment_code", "entities": ["TX-440"]},
            "query": "Which component failed when trains were held at Harbour Row?",
            "corruption": [{"op": "substitute", "relation": "caused_by", "source": "incident_2026_01",
                            "true_targets": ["equipment_TX-440"], "corrupt_targets": ["equipment_AX-3"]}]}
    result = score("AX-3.", item, "[1] (incident_2026_01) Trains were held at Harbour Row after a TX-440 trip.", OTHER_INCIDENT)
    assert result["outcome"] == "graph_misled"
    assert not result["graph_supported"]
    assert result["wrong_source"] == "text_only"


def test_gate_effect_labels():
    from experiments.robustness_r2 import gate_effect
    assert gate_effect("abstain", "graph_misled") == "blocked_wrong"
    assert gate_effect("abstain", "correct") == "blocked_correct"
    assert gate_effect("pass", "graph_misled") == "passed_wrong"
    assert gate_effect("pass", "correct") == "passed_correct"
    assert gate_effect("pass", "correct_abstention") == "passed_correct"
    assert gate_effect("abstain", "over_abstention") == "blocked_other"
