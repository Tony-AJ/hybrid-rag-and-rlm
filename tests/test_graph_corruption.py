"""Unit tests for experiments.graph_corruption (R2). Toy graph; no stores touched."""

import networkx as nx
import pytest

from experiments import graph_corruption as gc

TEXTS = {
    "incident_1": "Trains were held at Harbour Row after a TX-440 traction inverter tripped.",
    "equipment_TX-440": "The TX-440 traction inverter.",
    "equipment_BR-7": "The BR-7 regenerative brake unit.",
    "procedure_MP-101": "Maintenance procedure MP-101: Phase check. Applies to the TX-440 traction inverter.",
    "procedure_MP-105": "Maintenance procedure MP-105: Grid survey. Applies to the BR-7 regenerative brake unit.",
    "station_alpha": "Alpha is a terminus.", "station_bravo": "Bravo is a through station.",
    "station_charlie": "Charlie is a terminus.", "station_xray": "Xray is a terminus.",
    "line_one": "The Blue Line runs from Alpha to Charlie, calling at Bravo. It is operated with sets.",
}


def toy_graph():
    graph = nx.MultiDiGraph()
    for node, text in TEXTS.items():
        graph.add_node(node, text=text, metadata={}, embedding=None)
    edges = [("incident_1", "caused_by", "equipment_TX-440"),
             ("equipment_TX-440", "maintained_by", "procedure_MP-101"), ("procedure_MP-101", "maintains", "equipment_TX-440"),
             ("equipment_BR-7", "maintained_by", "procedure_MP-105"), ("procedure_MP-105", "maintains", "equipment_BR-7"),
             ("station_alpha", "next_station", "station_bravo"), ("station_bravo", "previous_station", "station_alpha"),
             ("station_bravo", "next_station", "station_charlie"), ("station_charlie", "previous_station", "station_bravo")]
    for source, relation, target in edges:
        graph.add_edge(source, target, id=f"{source}-{target}", type=relation, weight=1.0)
    return graph


class FakeGraphDB:
    def __init__(self, graph):
        self.graph = graph
        self.db_path = "toy.json"
        self.persisted = False

    def persist(self):
        self.persisted = True


class FakeService:
    def __init__(self):
        self.graph_db = FakeGraphDB(toy_graph())
        self.vector_db = object()
        self.graph_db_path = "toy.json"
        self.vector_db_path = "toy"

    def hybrid_search(self, *args):
        raise AssertionError("must not be reachable")


SUBSTITUTE = [{"op": "substitute", "relation": "maintained_by", "source": "equipment_TX-440",
               "true_targets": ["procedure_MP-101"], "corrupt_targets": ["procedure_MP-105"]}]


def test_substitute_moves_forward_and_reverse_edges_in_copy_only():
    service = FakeService()
    before = gc.edge_set(service.graph_db.graph)
    view = gc.CorruptedService(service, SUBSTITUTE)
    graph = view.graph_db.graph
    assert gc.edges_of(graph, "equipment_TX-440", "maintained_by") == ["procedure_MP-105"]
    assert "equipment_TX-440" in gc.edges_of(graph, "procedure_MP-105", "maintains")
    assert gc.edges_of(graph, "procedure_MP-101", "maintains") == []
    assert gc.edges_of(graph, "procedure_MP-105", "maintains") == ["equipment_BR-7", "equipment_TX-440"]
    assert gc.edge_set(service.graph_db.graph) == before
    assert not service.graph_db.persisted
    assert [entry["change"] for entry in view.change_log] == ["removed", "removed", "added", "added"]


def test_add_spurious_keeps_true_edges():
    view = gc.CorruptedService(FakeService(), [{"op": "add_spurious", "relation": "maintained_by",
                                                "source": "equipment_TX-440", "corrupt_targets": ["procedure_MP-105"]}])
    assert gc.edges_of(view.graph_db.graph, "equipment_TX-440", "maintained_by") == ["procedure_MP-101", "procedure_MP-105"]


def test_caused_by_has_no_reverse_edge():
    view = gc.CorruptedService(FakeService(), [{"op": "substitute", "relation": "caused_by", "source": "incident_1",
                                                "true_targets": ["equipment_TX-440"], "corrupt_targets": ["equipment_BR-7"]}])
    assert gc.edges_of(view.graph_db.graph, "incident_1", "caused_by") == ["equipment_BR-7"]


def test_next_station_substitution_updates_previous_station():
    view = gc.CorruptedService(FakeService(), [{"op": "substitute", "relation": "next_station", "source": "station_bravo",
                                                "true_targets": ["station_charlie"], "corrupt_targets": ["station_xray"]}])
    graph = view.graph_db.graph
    assert gc.edges_of(graph, "station_bravo", "next_station") == ["station_xray"]
    assert gc.edges_of(graph, "station_xray", "previous_station") == ["station_bravo"]
    assert gc.edges_of(graph, "station_charlie", "previous_station") == []


def test_view_get_node_reads_the_copy():
    view = gc.CorruptedService(FakeService(), SUBSTITUTE)
    assert view.graph_db.get_node("procedure_MP-101").text.startswith("Maintenance procedure MP-101")


@pytest.mark.parametrize("name", ["persist", "save", "add_edge", "create_node", "delete_node"])
def test_graph_view_refuses_persisting_calls(name):
    view = gc.CorruptedService(FakeService(), SUBSTITUTE)
    with pytest.raises(AttributeError):
        getattr(view.graph_db, name)


def test_service_view_refuses_hybrid_search():
    with pytest.raises(AttributeError):
        gc.CorruptedService(FakeService(), SUBSTITUTE).hybrid_search


def test_invalid_specs_raise():
    with pytest.raises(ValueError):
        gc.CorruptedService(FakeService(), [{"op": "substitute", "relation": "authorised_by", "source": "procedure_MP-101",
                                             "true_targets": [], "corrupt_targets": ["procedure_MP-105"]}])
    with pytest.raises(ValueError):
        gc.CorruptedService(FakeService(), [{"op": "substitute", "relation": "maintained_by", "source": "equipment_TX-440",
                                             "true_targets": ["procedure_MP-105"], "corrupt_targets": ["procedure_MP-101"]}])


def test_document_predicates():
    orders = gc.line_orders(TEXTS)
    assert orders == {"line_one": ["Alpha", "Bravo", "Charlie"]}
    assert gc.document_supports("maintained_by", "equipment_TX-440", "procedure_MP-101", TEXTS)
    assert not gc.document_supports("maintained_by", "equipment_TX-440", "procedure_MP-105", TEXTS)
    assert gc.document_supports("caused_by", "incident_1", "equipment_TX-440", TEXTS)
    assert not gc.document_supports("caused_by", "incident_1", "equipment_BR-7", TEXTS)
    assert gc.document_supports("next_station", "station_bravo", "station_charlie", TEXTS, orders)
    assert not gc.document_supports("next_station", "station_bravo", "station_xray", TEXTS, orders)


def test_empty_spec_is_identity():
    service = FakeService()
    view = gc.CorruptedService(service, [])
    assert gc.edge_set(view.graph_db.graph) == gc.edge_set(service.graph_db.graph)
    assert view.graph_db.graph is not service.graph_db.graph


def test_spec_hash_changes_with_spec():
    a = gc.spec_sha256([{"id": 1, "corruption": SUBSTITUTE}])
    b = gc.spec_sha256([{"id": 1, "corruption": []}])
    assert a != b
