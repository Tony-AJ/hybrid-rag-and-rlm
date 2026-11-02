"""Unit tests for rlm.robustness_scoring (R1). Pure functions, no stores, no model."""

import pytest

from rlm import robustness_scoring as r1s

UNIVERSE = ["Harbour Row", "Kestrel Green", "Alder Vale"]

STATION = ("[1] (station_harbour_row) Harbour Row is a terminus on the Azure Line with 3 platforms "
           "and step-free access to all platforms. The station building dates from 1902 and was last "
           "refurbished in 2009.")
EQUIPMENT = ("[1] (equipment_TX-440) The TX-440 traction inverter is fitted across the fleet. "
             "A TX-440 unit has a design life of 9 years. Links: maintained_by -> procedure_MP-101, "
             "procedure_MP-102")
PROCEDURE = ("[1] (procedure_MP-101) Maintenance procedure MP-101: Inverter phase balance verification. "
             "Scheduled every 90 days. Links: authorised_by -> policy_PL-A")


def prompt(*evidence_lines: str, question: str = "Q?") -> str:
    return "Answer using only the evidence.\nQuestion: " + question + "\nEvidence:\n" + "\n".join(evidence_lines) + "\nAnswer:"


R1A = {"id": 1001, "case": "R1a", "query": "Who is the station manager at Harbour Row?",
       "gold_answer": {"kind": "abstain"}}
R1B = {"id": 1009, "case": "R1b",
       "query": "How many platforms does Harbour Row have, and how many car parking spaces does it have?",
       "answerable_part": {"kind": "predicates", "predicates": {"platforms": 3}},
       "gold_answer": {"kind": "abstain"}}
R1B_CODE = {"id": 1013, "case": "R1b",
            "query": "How often is maintenance procedure MP-101 scheduled, and what does each visit cost?",
            "answerable_part": {"kind": "predicates", "predicates": {"interval_days": 90}},
            "gold_answer": {"kind": "abstain"}}
R1C = {"id": 1017, "case": "R1c",
       "query": "Which maintenance procedures cover the fault that stopped service at Harbour Row?",
       "gold_answer": {"kind": "abstain", "unsupported_entities": {
           "entity_type": "procedure_code", "entities": ["MP-101", "MP-102"]}}}


def score(answer, item, *lines):
    return r1s.score_r1(answer, item, [prompt(*lines, question=item["query"])], UNIVERSE)


def test_received_evidence_splits_text_and_links():
    evidence = r1s.received_evidence([prompt(EQUIPMENT)])
    assert "design life of 9 years" in evidence["text"]
    assert "MP-101" not in evidence["text"]
    assert "procedure_MP-101" in evidence["links"]
    assert evidence["node_ids"] == ["equipment_TX-440"]


def test_correct_abstention_strict():
    result = score("INSUFFICIENT EVIDENCE.", R1A, STATION)
    assert result["outcome"] == "correct_abstention"
    assert result["abstention_strict"] and result["correct_abstention_strict"]
    assert not result["contains_unsupported_claim"]


def test_correct_abstention_lenient_only():
    result = score("The evidence does not state who manages Harbour Row.", R1A, STATION)
    assert result["outcome"] == "correct_abstention"
    assert not result["abstention_strict"] and result["abstention_lenient"]
    assert not result["correct_abstention_strict"]


def test_partial_correct():
    result = score("Harbour Row has 3 platforms; the number of car parking spaces is not stated.", R1B, STATION)
    assert result["outcome"] == "partial_correct"
    assert result["answerable_part_correct"] is True


def test_partial_with_question_identifier_is_supported():
    result = score("MP-101 is scheduled every 90 days. Its cost is not stated in the evidence.", R1B_CODE, PROCEDURE)
    assert result["outcome"] == "partial_correct"
    assert result["named_identifiers"] == ["number:90"]


def test_link_only_answer():
    result = score("MP-101 and MP-102 cover the TX-440 fault.", R1C, EQUIPMENT)
    assert result["outcome"] == "link_only_answer"
    assert result["contains_unsupported_claim"]
    assert result["link_only_identifiers"] == ["code:MP-101", "code:MP-102"]
    assert result["masked_truth"]["named"] == ["MP-101", "MP-102"]
    assert result["masked_truth"]["exposed"] is True


def test_masked_truth_not_exposed_without_links():
    result = score("INSUFFICIENT EVIDENCE", R1C, STATION)
    assert result["masked_truth"]["exposed"] is False
    assert result["outcome"] == "correct_abstention"


def test_hedged_abstention_naming_link_target_is_not_an_abstention():
    result = score("INSUFFICIENT EVIDENCE, though the links point to MP-101.", R1C, EQUIPMENT)
    assert result["abstention_strict"]
    assert result["outcome"] == "link_only_answer"


def test_unsupported_answer():
    result = score("Harbour Row has 250 car parking spaces.", R1A, STATION)
    assert result["outcome"] == "unsupported_answer"
    assert result["unsupported_identifiers"] == ["number:250"]
    assert result["link_only_identifiers"] == []


def test_other_wrong():
    result = score("Harbour Row is run by the operations team.", R1A, STATION)
    assert result["outcome"] == "other_wrong"
    assert not result["contains_unsupported_claim"]


def test_r1b_answering_missing_half_is_unsupported():
    result = score("Harbour Row has 3 platforms and 120 parking spaces.", R1B, STATION)
    assert result["outcome"] == "unsupported_answer"
    assert result["answerable_part_correct"] is True


def test_draft_answer_in_grounding_prompt_is_not_support():
    grounding = ("Check the draft answer against the evidence.\nQuestion: q\nDraft answer: MP-101\n"
                 "Evidence:\n" + EQUIPMENT + "\nReply with the corrected answer.")
    result = r1s.score_r1("MP-101", R1C, [grounding], UNIVERSE)
    assert result["outcome"] == "link_only_answer"


def test_support_from_any_generation_call_counts():
    later = prompt("[2] (procedure_MP-101) Maintenance procedure MP-101: x. Scheduled every 90 days.")
    result = r1s.score_r1("MP-101", R1C, [prompt(EQUIPMENT), later], UNIVERSE)
    assert result["identifier_supported"]


def test_station_name_unsupported():
    result = score("Kestrel Green.", R1A, STATION)
    assert result["unsupported_identifiers"] == ["station:Kestrel Green"]


def test_dates_and_codes_are_not_numbers():
    assert r1s.identifiers("On 2026-01-04 the TX-440 failed", UNIVERSE) == {"code:TX-440"}


@pytest.mark.parametrize("outcome", r1s.OUTCOMES)
def test_every_outcome_class_is_reachable(outcome):
    examples = {
        "correct_abstention": ("INSUFFICIENT EVIDENCE", R1A, STATION),
        "partial_correct": ("3 platforms; parking is not stated.", R1B, STATION),
        "link_only_answer": ("MP-101", R1C, EQUIPMENT),
        "unsupported_answer": ("250 spaces", R1A, STATION),
        "other_wrong": ("Run by the operations team.", R1A, STATION),
    }
    answer, item, line = examples[outcome]
    assert score(answer, item, line)["outcome"] == outcome
