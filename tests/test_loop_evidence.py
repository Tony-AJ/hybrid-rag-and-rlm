"""Unit tests for rlm.loop_evidence (R1'). Fake retriever and scripted provider; no stores."""

from typing import Any, Dict, List, Optional

from rlm import loop as rlm_loop
from rlm import loop_evidence as ea
from rlm.providers import GenerationConfig, GenerationResult

UNIVERSE = ["Harbour Row", "Kestrel Green", "Alder Vale"]

NODES = {
    "incident_2026_01": ("Service incident 2026/01: trains were held at Harbour Row after a TX-440 "
                         "traction inverter suffered a phase imbalance trip.",
                         {"caused_by": ["equipment_TX-440"]}),
    "equipment_TX-440": ("The TX-440 traction inverter has a design life of 9 years.",
                         {"maintained_by": ["procedure_MP-101", "procedure_MP-102"]}),
    "procedure_MP-101": ("Maintenance procedure MP-101: Inverter phase balance verification.", {}),
    "procedure_MP-102": ("Maintenance procedure MP-102: Inverter gate driver replacement.", {}),
    "station_alder_vale": ("Alder Vale is a through station with 2 platforms.",
                           {"next_station": ["station_kestrel_green"]}),
    "station_kestrel_green": ("Kestrel Green is a through station with 2 platforms.", {}),
    "procedure_MP-999": ("", {}),
}


class FakeRetriever:
    def __init__(self, ranking: List[str], unavailable: Optional[List[str]] = None):
        self.ranking = ranking
        self.unavailable = set(unavailable or [])

    def chunk(self, node_id: str) -> Dict[str, Any]:
        text, relations = NODES[node_id]
        return {"node_id": node_id, "text": text, "hop": 0, "retrieved_by_vector": True, "relations": relations}

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        return [self.chunk(node_id) for node_id in self.ranking[:top_k]]

    def fetch(self, node_id: str) -> Optional[Dict[str, Any]]:
        return None if node_id in self.unavailable or node_id not in NODES else self.chunk(node_id)

    def relations(self, node_id: str) -> Dict[str, List[str]]:
        return NODES[node_id][1]


class ScriptedProvider:
    def __init__(self, replies: List[str]):
        self.replies = list(replies)
        self.prompts: List[str] = []
        self.config = GenerationConfig(provider="scripted", model="scripted", temperature=0.0, max_tokens=10)

    def generate(self, prompt: str) -> GenerationResult:
        self.prompts.append(prompt)
        return GenerationResult(text=self.replies.pop(0), model="scripted", provider="scripted",
                                input_tokens=len(prompt) // 4, output_tokens=5)


LEAN = rlm_loop.LoopBounds(max_steps=3, max_retrievals=3, top_k=5, grounding="lean")
QUESTION = "Which maintenance procedures cover the fault that stopped service at Harbour Row?"
EVIDENCE = ["incident_2026_01", "equipment_TX-440"]


def steps_of(run, kind):
    return [step for step in run.steps if step.get("step") == kind]


def test_unavailable_fetch_is_recorded_and_gated():
    retriever = FakeRetriever(EVIDENCE, unavailable=["procedure_MP-101", "procedure_MP-102"])
    provider = ScriptedProvider(["ANSWER: MP-101 and MP-102 cover the TX-440 fault."])
    run = ea.run_loop_ea(retriever, provider, QUESTION, LEAN, gate=True, abstain_option=True, universe=UNIVERSE)
    grounding = steps_of(run, "grounding")[0]
    assert grounding["fetched"] == []
    assert {f["reason"] for f in grounding["fetch_failures"]} == {"unavailable"}
    assert grounding["link_only_claims"] == ["procedure_MP-101", "procedure_MP-102"]
    gate = steps_of(run, "gate")[0]
    assert gate["decision"] == "abstain"
    assert gate["reasons"]["code:MP-101"] == "link_target_unavailable"
    assert run.answer == ea.ABSTAIN_ANSWER
    assert run.stop_reason == "gate_abstained"
    assert run.generation_calls == 1


def test_without_gate_failed_fetch_is_still_recorded_but_answer_kept():
    retriever = FakeRetriever(EVIDENCE, unavailable=["procedure_MP-101", "procedure_MP-102"])
    provider = ScriptedProvider(["ANSWER: MP-101 and MP-102."])
    run = ea.run_loop_ea(retriever, provider, QUESTION, LEAN, gate=False, abstain_option=True, universe=UNIVERSE)
    assert steps_of(run, "grounding")[0]["fetch_failures"]
    assert "MP-101" in run.answer


def test_gate_passes_when_endpoint_text_fetched():
    retriever = FakeRetriever(EVIDENCE)
    provider = ScriptedProvider(["ANSWER: MP-101 and MP-102.", "MP-101 and MP-102 cover the TX-440 fault."])
    run = ea.run_loop_ea(retriever, provider, QUESTION, LEAN, gate=True, abstain_option=True, universe=UNIVERSE)
    assert steps_of(run, "grounding")[0]["fetched"] == ["procedure_MP-101", "procedure_MP-102"]
    assert steps_of(run, "gate")[0]["decision"] == "pass"
    assert "MP-101" in run.answer


def test_adjacency_answer_passes_gate_when_neighbour_chunk_retrieved():
    retriever = FakeRetriever(["station_alder_vale", "station_kestrel_green"])
    provider = ScriptedProvider(["ANSWER: Kestrel Green."])
    run = ea.run_loop_ea(retriever, provider, "Which station is next to Alder Vale?", LEAN,
                         gate=True, abstain_option=True, universe=UNIVERSE)
    assert steps_of(run, "gate")[0]["decision"] == "pass"
    assert run.answer.startswith("Kestrel Green")


def test_station_named_only_through_a_link_is_not_fetched_and_is_gated():
    # Known deviation from the preregistration text: loop 1.7's cited_link_targets
    # matches codes, not station names, so grounding never fetches a neighbour
    # station. Measured on v1.1 adjacency (H3), not patched.
    retriever = FakeRetriever(["station_alder_vale"])
    provider = ScriptedProvider(["ANSWER: Kestrel Green."])
    run = ea.run_loop_ea(retriever, provider, "Which station is next to Alder Vale?", LEAN,
                         gate=True, abstain_option=True, universe=UNIVERSE)
    assert not steps_of(run, "grounding")
    assert steps_of(run, "gate")[0]["reasons"] == {"station:Kestrel Green": "not_in_received_text"}


def test_empty_text_fetch_is_a_failure():
    NODES["equipment_TX-440"] = (NODES["equipment_TX-440"][0], {"maintained_by": ["procedure_MP-999"]})
    try:
        retriever = FakeRetriever(EVIDENCE)
        provider = ScriptedProvider(["ANSWER: MP-999."])
        run = ea.run_loop_ea(retriever, provider, QUESTION, LEAN, gate=True, abstain_option=True, universe=UNIVERSE)
        assert steps_of(run, "grounding")[0]["fetch_failures"] == [{"node_id": "procedure_MP-999", "reason": "empty_text"}]
        assert run.answer == ea.ABSTAIN_ANSWER
    finally:
        NODES["equipment_TX-440"] = (NODES["equipment_TX-440"][0],
                                     {"maintained_by": ["procedure_MP-101", "procedure_MP-102"]})


def test_truncated_citations_are_recorded():
    bounds = rlm_loop.LoopBounds(max_steps=3, top_k=5, grounding="lean", max_grounding_chunks=1)
    retriever = FakeRetriever(EVIDENCE)
    provider = ScriptedProvider(["ANSWER: MP-101 and MP-102.", "MP-101 and MP-102."])
    run = ea.run_loop_ea(retriever, provider, QUESTION, bounds, gate=True, abstain_option=True, universe=UNIVERSE)
    grounding = steps_of(run, "grounding")[0]
    assert grounding["truncated"] == ["procedure_MP-102"]
    assert {"node_id": "procedure_MP-102", "reason": "truncated"} in grounding["fetch_failures"]
    assert steps_of(run, "gate")[0]["decision"] == "abstain"


def test_abstain_decision_is_parsed():
    retriever = FakeRetriever(["incident_2026_01"])
    provider = ScriptedProvider(["ABSTAIN: no staff names in the evidence."])
    run = ea.run_loop_ea(retriever, provider, "Who manages Harbour Row?", LEAN, gate=True,
                         abstain_option=True, universe=UNIVERSE)
    assert run.answer == ea.ABSTAIN_ANSWER
    assert run.stop_reason == "model_abstained"
    assert run.steps[0]["abstain_reason"].startswith("no staff names")
    assert run.generation_calls == 1
    assert "ABSTAIN:" in provider.prompts[0]


def test_abstain_option_off_uses_loop17_step_template():
    retriever = FakeRetriever(["incident_2026_01"])
    provider = ScriptedProvider(["ANSWER: TX-440.", "TX-440."])
    ea.run_loop_ea(retriever, provider, QUESTION, LEAN, gate=False, abstain_option=False, universe=UNIVERSE)
    assert "ABSTAIN:" not in provider.prompts[0]


def test_equivalence_with_loop17_when_gate_and_abstain_off():
    scripts = [["ANSWER: MP-101 and MP-102.", "MP-101 and MP-102."],
               ["SEARCH: procedures for TX-440", "ANSWER: MP-101.", "MP-101."],
               ["SEARCH: a", "SEARCH: b", "INSUFFICIENT EVIDENCE"]]
    for script in scripts:
        outcomes = []
        for runner in ("1.7", "ea"):
            provider = ScriptedProvider(script)
            retriever = FakeRetriever(EVIDENCE)
            if runner == "1.7":
                run = rlm_loop.run_loop(retriever, provider, QUESTION, LEAN)
            else:
                run = ea.run_loop_ea(retriever, provider, QUESTION, LEAN, gate=False, abstain_option=False,
                                     universe=UNIVERSE)
            outcomes.append((provider.prompts, run.answer, run.evidence_ids, run.generation_calls,
                             run.input_tokens, run.stop_reason))
        assert outcomes[0] == outcomes[1]


def test_gate_ignores_identifiers_named_in_question():
    verdict = ea.sufficiency_gate("TX-440 failed at Harbour Row.", "Which component failed at Harbour Row?",
                                  ["The TX-440 traction inverter."], UNIVERSE, [])
    assert verdict["decision"] == "pass"
    verdict = ea.sufficiency_gate("Kestrel Green.", "Which station?", ["Alder Vale is a station."], UNIVERSE, [])
    assert verdict["decision"] == "abstain"
    assert verdict["reasons"] == {"station:Kestrel Green": "not_in_received_text"}


def test_gate_does_not_gate_numbers():
    verdict = ea.sufficiency_gate("It has 250 spaces.", "How many spaces?", ["No numbers here."], UNIVERSE, [])
    assert verdict["decision"] == "pass"
