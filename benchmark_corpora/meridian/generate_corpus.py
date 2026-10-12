"""Deterministic generator for the Meridian Transit benchmark corpus (v1).

Why this corpus exists
----------------------
The Romeo and Juliet corpus has 8 chunks, so at K=5 vector search already sees
62% of it and no fusion strategy has room to change the ranking. This corpus is
built to make graph evidence *decidable*:

  - 160 chunks, of which ~19% are deliberate distractors, so precision@K can
    actually move.
  - A typed directed graph whose edge direction matches the production
    expansion, which follows `graph.successors` only
    (app/service.py: hybrid_search step 2).
  - Query families with a declared hop distance. The multi-hop families are
    written so the answer chunk shares little vocabulary with the query, while a
    chunk that vector search *does* find sits 1-2 outgoing hops away from it.

Ground truth is correct by construction: every label is read off the same
entity tables that produced the text and the edges. Nothing is hand-judged.

What this does NOT claim: the text is synthetic, so results measure whether the
fusion mechanism can exploit graph structure, not how the system behaves on
natural prose.

Run:
    python benchmark_corpora/meridian/generate_corpus.py

Writes (all beside this file):
    corpus_documents.json  - [{id, text, metadata}]
    corpus_graph.json      - {nodes, edges} in graph_db JSON format
    dataset.json           - {corpus_id, dataset_version, queries: [...]}

It never touches an existing vector index or graph. Ingestion is a separate
step (ingest.py).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
CORPUS_ID = "meridian-transit-v1"
DATASET_VERSION = "1.1"
SEED = 20261012

# ============================== Entity tables ===============================

LINES: List[Tuple[str, str, str]] = [
    # (line key, public name, rolling stock family)
    ("azure", "Azure Line", "Series 400 four-car sets"),
    ("crimson", "Crimson Line", "Series 500 six-car sets"),
    ("jade", "Jade Line", "Series 300 refurbished sets"),
    ("amber", "Amber Line", "Series 600 articulated sets"),
    ("violet", "Violet Line", "Series 450 driverless sets"),
    ("slate", "Slate Line", "Series 200 heritage sets"),
]

# 36 stations, six per line, in running order.
STATIONS: Dict[str, List[str]] = {
    "azure": ["Harbour Row", "Kestrel Green", "Alder Vale", "Brickworks", "Copper Lane", "Dunmore Park"],
    "crimson": ["Eastgate Quay", "Foundry Bridge", "Garnet Hill", "Hollow Mill", "Ironside", "Jubilee Fields"],
    "jade": ["Kingsmoor", "Larkspur", "Mallow Street", "Netherby", "Oakhurst", "Pennyford"],
    "amber": ["Quarry Side", "Redcliff", "Saltmarsh", "Thistlebank", "Underbank", "Vantage Point"],
    "violet": ["Westmere", "Yarrow Cross", "Ashfen", "Beacon Rise", "Carrow End", "Dovecote"],
    "slate": ["Elmsworth", "Fallowfield", "Glasshouse", "Hartsmere", "Ivory Dock", "Juniper Row"],
}

# (code, component name, characteristic fault phrase, diagnostic signature)
EQUIPMENT: List[Tuple[str, str, str, str]] = [
    ("TX-440", "traction inverter", "phase imbalance trip", "unbalanced current draw across the three output phases"),
    ("HV-21", "pantograph head", "carbon strip fracture", "a sudden loss of overhead contact force"),
    ("BR-7", "regenerative brake unit", "resistor grid overheat", "grid temperature above the 310 degree cut-out"),
    ("PD-90", "platform screen door controller", "obstruction sensor drift", "false obstruction reports on an empty threshold"),
    ("SG-12", "signal interlocking relay", "contact welding", "a relay that fails to drop out after de-energising"),
    ("AX-3", "axle bearing assembly", "thermal spike", "a bearing temperature rise of more than 40 degrees in ten minutes"),
    ("CP-55", "compressor module", "pressure bleed-down", "main reservoir pressure falling below 6.2 bar while stabled"),
    ("DR-8", "door engine actuator", "leadscrew backlash", "a door leaf that rebounds before locking"),
    ("PS-19", "auxiliary power supply", "ripple fault", "excess ripple voltage on the 110 volt battery bus"),
    ("TC-6", "track circuit transmitter", "frequency drift", "a carrier frequency outside the 83 to 87 hertz window"),
    ("VE-31", "tunnel ventilation fan", "blade imbalance", "vibration velocity above 7 millimetres per second at the bearing housing"),
    ("ES-14", "escalator drive chain", "elongation wear", "chain elongation beyond the 1.5 percent service limit"),
    ("LT-77", "lighting ballast", "flicker cascade", "a cascading flicker across an entire platform circuit"),
    ("WS-5", "wheel slide protection valve", "latched release", "a brake release that stays latched after the slide event clears"),
]

# (procedure code, equipment code, title, interval, supersedes or None)
PROCEDURES: List[Tuple[str, str, str, str, Any]] = [
    ("MP-101", "TX-440", "Inverter phase balance verification", "every 90 days", None),
    ("MP-102", "TX-440", "Inverter gate driver replacement", "on condition", "MP-101"),
    ("MP-103", "HV-21", "Pantograph carbon strip renewal", "every 45 days", None),
    ("MP-104", "HV-21", "Pantograph contact force calibration", "every 180 days", None),
    ("MP-105", "BR-7", "Brake resistor grid thermal survey", "every 60 days", None),
    ("MP-106", "BR-7", "Brake chopper firmware refresh", "annually", "MP-105"),
    ("MP-107", "PD-90", "Screen door sensor realignment", "every 30 days", None),
    ("MP-108", "PD-90", "Screen door controller loop test", "every 120 days", None),
    ("MP-109", "SG-12", "Interlocking relay contact inspection", "every 28 days", None),
    ("MP-110", "SG-12", "Relay drop-out timing audit", "every 365 days", None),
    ("MP-111", "AX-3", "Axle bearing lubrication and endplay check", "every 75 days", None),
    ("MP-112", "AX-3", "Bearing acoustic emission screening", "every 150 days", "MP-111"),
    ("MP-113", "CP-55", "Compressor valve plate overhaul", "every 200 days", None),
    ("MP-114", "CP-55", "Reservoir leak-rate measurement", "every 50 days", None),
    ("MP-115", "DR-8", "Door actuator leadscrew re-shimming", "every 40 days", None),
    ("MP-116", "DR-8", "Door obstacle force profiling", "every 90 days", None),
    ("MP-117", "PS-19", "Auxiliary supply ripple measurement", "every 70 days", None),
    ("MP-118", "TC-6", "Track circuit carrier retune", "every 35 days", None),
    ("MP-119", "TC-6", "Transmitter oscillator substitution", "on condition", "MP-118"),
    ("MP-120", "VE-31", "Ventilation fan balance correction", "every 240 days", None),
    ("MP-121", "ES-14", "Escalator chain elongation gauging", "every 110 days", None),
    ("MP-122", "LT-77", "Ballast circuit staged replacement", "every 300 days", None),
    ("MP-123", "WS-5", "Slide protection valve latch release test", "every 55 days", None),
    ("MP-124", "WS-5", "Wheel slide event log reconciliation", "every 25 days", "MP-123"),
]

# (policy code, title, scope sentence, procedures authorised)
POLICIES: List[Tuple[str, str, str, List[str]]] = [
    ("PL-A", "Traction Maintenance Authorisation", "work on propulsion and current collection assets", ["MP-101", "MP-102", "MP-103", "MP-104"]),
    ("PL-B", "Braking Systems Authorisation", "work that alters braking performance", ["MP-105", "MP-106", "MP-123", "MP-124"]),
    ("PL-C", "Platform Interface Authorisation", "work on the platform-train interface", ["MP-107", "MP-108"]),
    ("PL-D", "Signalling Safety Authorisation", "work inside a signalling interlocking", ["MP-109", "MP-110", "MP-118", "MP-119"]),
    ("PL-E", "Running Gear Authorisation", "work on axles, bearings and wheelsets", ["MP-111", "MP-112"]),
    ("PL-F", "Pneumatics Authorisation", "work on compressed air generation and storage", ["MP-113", "MP-114"]),
    ("PL-G", "Door Systems Authorisation", "work on vehicle door mechanisms", ["MP-115", "MP-116"]),
    ("PL-H", "Low Voltage Authorisation", "work on battery and auxiliary supplies", ["MP-117"]),
    ("PL-I", "Tunnel Environment Authorisation", "work on tunnel ventilation and smoke control", ["MP-120"]),
    ("PL-J", "Fixed Plant Authorisation", "work on escalators, lifts and platform lighting", ["MP-121", "MP-122"]),
    ("PL-K", "Night Possession Authorisation", "any procedure performed inside an engineering possession", ["MP-109", "MP-118", "MP-120", "MP-121"]),
    ("PL-L", "Degraded Mode Authorisation", "procedures carried out while service is running", ["MP-107", "MP-114", "MP-124"]),
]

# (role key, job title, procedures owned)
ROLES: List[Tuple[str, str, List[str]]] = [
    ("traction_technician", "night traction technician", ["MP-101", "MP-102"]),
    ("current_collection_fitter", "current collection fitter", ["MP-103", "MP-104"]),
    ("brake_specialist", "brake systems specialist", ["MP-105", "MP-106"]),
    ("platform_systems_engineer", "platform systems engineer", ["MP-107", "MP-108"]),
    ("signalling_technician", "signalling maintenance technician", ["MP-109", "MP-110", "MP-118", "MP-119"]),
    ("running_gear_inspector", "running gear inspector", ["MP-111", "MP-112"]),
    ("pneumatics_fitter", "pneumatics fitter", ["MP-113", "MP-114"]),
    ("door_fitter", "door systems fitter", ["MP-115", "MP-116"]),
    ("power_technician", "auxiliary power technician", ["MP-117", "MP-122"]),
    ("plant_engineer", "fixed plant engineer", ["MP-120", "MP-121", "MP-123", "MP-124"]),
]

# Distractor themes: plausible authority documents with no maintenance content.
BULLETIN_THEMES: List[Tuple[str, str]] = [
    ("fare", "Fare and ticketing bulletin"),
    ("accessibility", "Accessibility works notice"),
    ("lost_property", "Lost property quarterly summary"),
    ("staff_news", "Staff newsletter extract"),
    ("timetable", "Timetable consultation summary"),
    ("retail", "Station retail licensing update"),
    ("cycling", "Cycle parking expansion notice"),
    ("catering", "Depot catering arrangements"),
    ("uniform", "Uniform and presentation standard"),
    ("archive", "Heritage archive acquisition note"),
]


# ============================ Derived structures ============================


def station_id(name: str) -> str:
    return "station_" + name.lower().replace(" ", "_")


def line_id(key: str) -> str:
    return f"line_{key}"


def equipment_id(code: str) -> str:
    return f"equipment_{code}"


def procedure_id(code: str) -> str:
    return f"procedure_{code}"


def policy_id(code: str) -> str:
    return f"policy_{code}"


def role_id(key: str) -> str:
    return f"role_{key}"


def incident_id(number: int) -> str:
    return f"incident_2026_{number:02d}"


def bulletin_id(number: int) -> str:
    return f"bulletin_{number:02d}"


def build_incidents() -> List[Dict[str, Any]]:
    """28 incidents, each pinned to one station and one equipment fault.

    Assignment is index arithmetic, not randomness, so the station -> equipment
    mapping is stable and auditable. Stations are taken in running order across
    lines so that every line carries incidents.
    """
    flat_stations = [
        (key, name) for key, names in STATIONS.items() for name in names
    ]
    incidents: List[Dict[str, Any]] = []
    for number in range(1, 29):
        line_key, station_name = flat_stations[(number - 1) % len(flat_stations)]
        code, component, fault, _signature = EQUIPMENT[(number - 1) % len(EQUIPMENT)]
        incidents.append({
            "id": incident_id(number),
            "number": number,
            "line_key": line_key,
            "station": station_name,
            "equipment_code": code,
            "component": component,
            "fault": fault,
            "date": f"2026-{((number - 1) % 12) + 1:02d}-{((number * 3) % 27) + 1:02d}",
            "delay_minutes": 7 + (number * 4) % 53,
            "trains_held": 2 + (number % 9),
        })
    return incidents


def procedures_for_equipment(code: str) -> List[str]:
    return [proc[0] for proc in PROCEDURES if proc[1] == code]


def policies_for_procedure(code: str) -> List[str]:
    return [policy[0] for policy in POLICIES if code in policy[3]]


def roles_for_procedure(code: str) -> List[str]:
    return [role[0] for role in ROLES if code in role[2]]


# ================================ Documents =================================


def build_documents(incidents: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []

    def add(doc_id: str, text: str, doc_type: str, **extra: Any) -> None:
        metadata = {"id": doc_id, "type": doc_type, "corpus": CORPUS_ID}
        metadata.update({key: value for key, value in extra.items() if value is not None})
        documents.append({"id": doc_id, "text": " ".join(text.split()), "metadata": metadata})

    # --- Lines ---
    for key, public_name, stock in LINES:
        names = STATIONS[key]
        add(
            line_id(key),
            f"""The {public_name} runs from {names[0]} to {names[-1]}, calling at
            {', '.join(names[1:-1])}. It is operated with {stock} and carries an
            average weekday ridership of {38 + len(key) * 7} thousand journeys.
            Headways are {3 + len(names) % 4} minutes in the morning peak and
            {8 + len(names) % 5} minutes in the evening off-peak.""",
            "line",
            line=public_name,
        )

    # --- Stations ---
    for key, public_name, _stock in LINES:
        names = STATIONS[key]
        for position, name in enumerate(names):
            role = "terminus" if position in (0, len(names) - 1) else "through station"
            platforms = 2 if role == "through station" else 3
            add(
                station_id(name),
                f"""{name} is a {role} on the {public_name} with {platforms} platforms
                and step-free access to {'all' if position % 2 == 0 else 'the northbound'}
                platforms. It is the {position + 1} stop counted from {names[0]} and
                records about {11 + position * 3} thousand entries on a typical weekday.
                The station building dates from {1902 + position * 7} and was last
                refurbished in {2009 + position}.""",
                "station",
                line=public_name,
                position=position + 1,
            )

    # --- Equipment ---
    for code, component, fault, signature in EQUIPMENT:
        add(
            equipment_id(code),
            f"""The {code} {component} is fitted across the fleet and in fixed
            installations. Its dominant failure mode is {fault}, which presents as
            {signature}. A {code} unit has a design life of
            {8 + (len(code) % 5)} years and is classified as a
            {'safety-critical' if code[0] in 'BSTAW' else 'service-critical'} asset.
            Replacement units are held at the central store under part group
            {code.replace('-', '')}G.""",
            "equipment",
            equipment_code=code,
        )

    # --- Procedures: deliberately never name a station or a line ---
    for code, equipment_code, title, interval, supersedes in PROCEDURES:
        component = next(item[1] for item in EQUIPMENT if item[0] == equipment_code)
        steps = rng.choice([
            "isolate the unit, record the as-found reading, carry out the corrective step, then prove the unit through two full cycles",
            "apply the local isolation, measure against the acceptance table, replace any part outside limits, and seal the enclosure",
            "take the asset out of service, complete the measurement sheet, restore the configuration, and log the result in the asset register",
        ])
        supersede_clause = (
            f" This procedure supersedes {supersedes}, which is withdrawn from the maintenance plan."
            if supersedes else ""
        )
        add(
            procedure_id(code),
            f"""Maintenance procedure {code}: {title}. Applies to the {equipment_code}
            {component}. Scheduled {interval}. Method: {steps}. Two competent staff are
            required and the work takes about {40 + (int(code[3:]) % 7) * 15} minutes.
            {supersede_clause}""",
            "procedure",
            procedure_code=code,
            equipment_code=equipment_code,
        )

    # --- Policies ---
    for code, title, scope, authorised in POLICIES:
        add(
            policy_id(code),
            f"""Authorisation policy {code}, {title}, governs {scope}. A named holder of
            {code} must countersign the work record before the asset is returned to
            service. The policy was issued in 2024, reviewed in 2026, and covers
            {len(authorised)} maintenance procedures. Withdrawal of {code} suspends
            every procedure it covers until a new holder is appointed.""",
            "policy",
            policy_code=code,
        )

    # --- Roles ---
    for key, title, owned in ROLES:
        add(
            role_id(key),
            f"""The {title} owns {len(owned)} scheduled maintenance procedures and
            reports to the engineering duty manager. The post requires
            {2 + len(owned)} years of depot experience, a current track safety
            certificate, and annual reassessment. Rostering places the post on
            {'nights' if 'night' in title else 'rotating shifts'}.""",
            "role",
            role=title,
        )

    # --- Incidents: name the station, never the procedure ---
    for incident in incidents:
        public_name = next(item[1] for item in LINES if item[0] == incident["line_key"])
        add(
            incident["id"],
            f"""Service incident {incident['id'].replace('incident_', '').replace('_', '/')}
            on {incident['date']}: trains were held at {incident['station']} on the
            {public_name} for {incident['delay_minutes']} minutes after a
            {incident['equipment_code']} {incident['component']} suffered a
            {incident['fault']}. {incident['trains_held']} trains were held behind the
            failure and the station was placed in degraded mode until the unit was
            isolated. Customers were advised to use alternative routes.""",
            "incident",
            station=incident["station"],
            line=public_name,
            equipment_code=incident["equipment_code"],
        )

    # --- Distractors: plausible authority prose, no maintenance content ---
    number = 0
    for theme_key, theme_title in BULLETIN_THEMES:
        for variant in range(3):
            number += 1
            quarter = (variant % 4) + 1
            add(
                bulletin_id(number),
                f"""{theme_title}, quarter {quarter} of 2026, reference B{number:03d}.
                The authority confirms that {theme_key.replace('_', ' ')} arrangements
                were reviewed by the customer committee, with
                {12 + number * 2} responses received and
                {3 + number % 6} actions carried forward. Nothing in this bulletin
                alters engineering standards, maintenance intervals or the
                authorisation framework. Enquiries go to the communications desk.""",
                "bulletin",
                theme=theme_key,
            )

    return documents


# ================================== Graph ===================================


def build_graph(
    documents: List[Dict[str, Any]],
    incidents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Typed directed graph.

    Edge direction is chosen so that the chunk vector search can find points
    *outwards* to the chunk it should pull in, because production expansion
    follows `graph.successors` only. Each multi-hop query family below names the
    path it depends on.
    """
    by_id = {doc["id"]: doc for doc in documents}
    nodes = [
        {"id": doc["id"], "text": doc["text"], "metadata": doc["metadata"], "embedding": None}
        for doc in documents
    ]
    edges: List[Dict[str, Any]] = []
    counter = 0

    def add_edge(source: str, target: str, edge_type: str, weight: float = 1.0) -> None:
        nonlocal counter
        assert source in by_id, f"edge source not a document: {source}"
        assert target in by_id, f"edge target not a document: {target}"
        counter += 1
        edges.append({
            "id": f"e{counter:04d}",
            "source": source,
            "target": target,
            "type": edge_type,
            "weight": weight,
        })

    # line <-> station, both directions (either side can be the vector seed)
    for key, _public_name, _stock in LINES:
        names = STATIONS[key]
        for position, name in enumerate(names):
            add_edge(line_id(key), station_id(name), "serves")
            add_edge(station_id(name), line_id(key), "served_by")
            if position + 1 < len(names):
                add_edge(station_id(name), station_id(names[position + 1]), "next_station")
                add_edge(station_id(names[position + 1]), station_id(name), "previous_station")

    # incident -> station / line (provenance) and incident -> equipment (cause)
    for incident in incidents:
        add_edge(incident["id"], station_id(incident["station"]), "occurred_at")
        add_edge(incident["id"], line_id(incident["line_key"]), "affected_line")
        add_edge(incident["id"], equipment_id(incident["equipment_code"]), "caused_by")
        add_edge(station_id(incident["station"]), incident["id"], "site_of_incident")

    # equipment -> procedure (hop 2 of the headline multi-hop path)
    for code, _component, _fault, _signature in EQUIPMENT:
        for proc_code in procedures_for_equipment(code):
            add_edge(equipment_id(code), procedure_id(proc_code), "maintained_by")
            add_edge(procedure_id(proc_code), equipment_id(code), "maintains")

    # procedure -> policy / role, and supersession chains
    for code, _equipment_code, _title, _interval, supersedes in PROCEDURES:
        for policy_code in policies_for_procedure(code):
            add_edge(procedure_id(code), policy_id(policy_code), "authorised_by")
            add_edge(policy_id(policy_code), procedure_id(code), "authorises")
        for key in roles_for_procedure(code):
            add_edge(procedure_id(code), role_id(key), "performed_by")
            add_edge(role_id(key), procedure_id(code), "responsible_for")
        if supersedes:
            add_edge(procedure_id(code), procedure_id(supersedes), "supersedes")
            add_edge(procedure_id(supersedes), procedure_id(code), "superseded_by")

    return {"nodes": nodes, "edges": edges}


# ================================= Dataset ==================================


def build_queries(incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Query families with declared hop distance and declared graph path.

    `relevant` is read off the entity tables, never hand-judged. `graph_path`
    records the outgoing path from a chunk vector search can plausibly find to
    the labelled answer, so an ablation can check whether that path was used.
    """
    queries: List[Dict[str, Any]] = []

    def add(
        query: str,
        relevant: List[str],
        category: str,
        hops: int,
        graph_path: str,
        description: str,
        gold_answer: Dict[str, Any],
    ) -> None:
        queries.append({
            "id": len(queries) + 1,
            "query": query,
            "relevant": relevant,
            "category": category,
            "hops": hops,
            "graph_path": graph_path,
            "description": description,
            # Answer-level ground truth, kept apart from the chunk-level
            # `relevant` labels. Two shapes only:
            #   entity_set - an authoritative set of identifiers, compared as a
            #                set after type-specific extraction from the answer
            #   predicates - named facts for free-form answers, each checked by
            #                a normalized value match, because exact string
            #                equality over prose is not a meaningful metric
            "gold_answer": gold_answer,
        })

    flat_stations = [(key, name) for key, names in STATIONS.items() for name in names]

    # --- Family 1: single-hop factual, 20 queries. Guards against regression. ---
    for index in range(12):
        line_key, name = flat_stations[index * 3 % len(flat_stations)]
        public_name = next(item[1] for item in LINES if item[0] == line_key)
        stop_position = STATIONS[line_key].index(name)
        is_terminus = stop_position in (0, len(STATIONS[line_key]) - 1)
        add(
            f"How many platforms does {name} have and when was the station building refurbished?",
            [station_id(name)],
            "single-hop factual",
            0,
            "none - answer chunk is the lexical match",
            f"{name} station chunk states platform count and refurbishment year.",
            {
                "kind": "predicates",
                "predicates": {
                    "platforms": 3 if is_terminus else 2,
                    "refurbished_year": 2009 + stop_position,
                },
            },
        )
        if len(queries) >= 12:
            break
    for key, public_name, _stock in LINES:
        add(
            f"What rolling stock operates the {public_name} and what are its peak headways?",
            [line_id(key)],
            "single-hop factual",
            0,
            "none - answer chunk is the lexical match",
            f"{public_name} chunk states stock family and headways.",
            {
                "kind": "predicates",
                "predicates": {
                    "rolling_stock": _stock,
                    "peak_headway_minutes": 3 + len(STATIONS[key]) % 4,
                },
            },
        )
    for code, _title, _scope, _authorised in POLICIES[:2]:
        add(
            f"What does authorisation policy {code} require before an asset returns to service?",
            [policy_id(code)],
            "single-hop factual",
            0,
            "none - answer chunk is the lexical match",
            f"Policy {code} chunk states the countersignature requirement.",
            {
                "kind": "predicates",
                "predicates": {
                    "policy_code": code,
                    "requirement": "countersign",
                    "signatory": "named holder",
                },
            },
        )

    # --- Family 2: 1 hop, incident -> equipment, 10 queries. ---
    for incident in incidents[:10]:
        add(
            f"Which component failed when trains were held at {incident['station']}?",
            [equipment_id(incident["equipment_code"])],
            "1-hop graph (cause)",
            1,
            f"{incident['id']} -caused_by-> {equipment_id(incident['equipment_code'])}",
            (
                f"The query names only the station. The equipment chunk never names a "
                f"station, so it must be reached from the incident chunk."
            ),
            {
                "kind": "entity_set",
                "entity_type": "equipment_code",
                "entities": [incident["equipment_code"]],
            },
        )

    # --- Family 3: 2 hops, incident -> equipment -> procedure, 20 queries. ---
    # The headline test: the labelled procedure chunk contains no station or
    # line vocabulary at all, so lexical/embedding similarity to the query is weak.
    for incident in incidents[:20]:
        proc_codes = procedures_for_equipment(incident["equipment_code"])
        add(
            f"Which maintenance procedures cover the fault that stopped service at {incident['station']}?",
            [procedure_id(code) for code in proc_codes],
            "2-hop graph (incident to procedure)",
            2,
            (
                f"{incident['id']} -caused_by-> {equipment_id(incident['equipment_code'])} "
                f"-maintained_by-> {', '.join(procedure_id(code) for code in proc_codes)}"
            ),
            (
                "Procedure chunks name only the equipment code, never the station, so "
                "vector search has no lexical bridge from the query to the answer."
            ),
            {
                "kind": "entity_set",
                "entity_type": "procedure_code",
                "entities": list(proc_codes),
            },
        )

    # --- Family 4: 1 hop, procedure -> policy, 6 queries. ---
    for code, equipment_code, title, _interval, _supersedes in PROCEDURES[:6]:
        policy_codes = policies_for_procedure(code)
        add(
            f"Which authorisation is needed to carry out {title.lower()} on the {equipment_code}?",
            [policy_id(policy_code) for policy_code in policy_codes],
            "1-hop graph (authorisation)",
            1,
            f"{procedure_id(code)} -authorised_by-> {', '.join(policy_id(p) for p in policy_codes)}",
            "Policy chunks never name a procedure title, only a scope phrase.",
            {
                "kind": "entity_set",
                "entity_type": "policy_code",
                "entities": list(policy_codes),
            },
        )

    # --- Family 5: 1 hop, next/previous station, 4 queries. ---
    for key in ["azure", "crimson", "jade", "amber"]:
        names = STATIONS[key]
        middle = names[2]
        add(
            f"Which stations sit immediately either side of {middle} along its own line?",
            [station_id(names[1]), station_id(names[3])],
            "1-hop graph (adjacency)",
            1,
            f"{station_id(middle)} -next_station/previous_station-> neighbours",
            "Adjacency is only in the graph; station text gives a stop number, not neighbour names.",
            {
                "kind": "entity_set",
                "entity_type": "station_name",
                "entities": [names[1], names[3]],
            },
        )

    return queries


# ================================== Writer ==================================


def validate(
    documents: List[Dict[str, Any]],
    graph: Dict[str, Any],
    queries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Refuse to emit a corpus whose own labels do not hold."""
    doc_ids = [doc["id"] for doc in documents]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("duplicate document ids generated")

    node_ids = {node["id"] for node in graph["nodes"]}
    if node_ids != set(doc_ids):
        raise ValueError("graph nodes and documents disagree")

    # Outgoing adjacency, to prove each labelled answer is reachable within its
    # declared hop count from the seed named in graph_path.
    successors: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    for edge in graph["edges"]:
        successors[edge["source"]].append(edge["target"])

    def reachable(seed: str, depth: int) -> set:
        frontier = {seed}
        seen = {seed}
        for _ in range(depth):
            nxt = set()
            for node_id in frontier:
                for target in successors[node_id]:
                    if target not in seen:
                        seen.add(target)
                        nxt.add(target)
            frontier = nxt
        return seen

    unreachable: List[str] = []
    for query in queries:
        for node_id in query["relevant"]:
            if node_id not in node_ids:
                raise ValueError(f"query {query['id']} labels unknown id {node_id}")
        if query["hops"] == 0:
            continue
        seed = query["graph_path"].split(" -")[0].strip()
        if seed not in node_ids:
            raise ValueError(f"query {query['id']} graph_path names unknown seed {seed}")
        hull = reachable(seed, query["hops"])
        missing = [node_id for node_id in query["relevant"] if node_id not in hull]
        if missing:
            unreachable.append(f"query {query['id']}: {missing} not within {query['hops']} hops of {seed}")
    if unreachable:
        raise ValueError("graph does not support the declared labels:\n" + "\n".join(unreachable))

    gold_kinds: Dict[str, int] = {}
    documents_by_id = {doc["id"]: doc for doc in documents}
    for query in queries:
        gold = query.get("gold_answer")
        if not isinstance(gold, dict) or "kind" not in gold:
            raise ValueError(f"query {query['id']} has no well-formed gold_answer")
        kind = gold["kind"]
        gold_kinds[kind] = gold_kinds.get(kind, 0) + 1
        if kind == "entity_set":
            if not gold.get("entities") or not gold.get("entity_type"):
                raise ValueError(f"query {query['id']} has an empty entity_set gold_answer")
            # Every gold entity must be stated in at least one labelled chunk,
            # otherwise the answer is not supported by the evidence we score.
            for entity in gold["entities"]:
                supported = any(
                    entity in documents_by_id[node_id]["text"]
                    for node_id in query["relevant"]
                )
                if not supported:
                    raise ValueError(
                        f"query {query['id']} gold entity {entity!r} appears in none of its "
                        f"labelled chunks {query['relevant']}"
                    )
        elif kind == "predicates":
            if not gold.get("predicates"):
                raise ValueError(f"query {query['id']} has an empty predicates gold_answer")
            for key, value in gold["predicates"].items():
                supported = any(
                    str(value).lower() in documents_by_id[node_id]["text"].lower()
                    for node_id in query["relevant"]
                )
                if not supported:
                    raise ValueError(
                        f"query {query['id']} predicate {key}={value!r} is not stated in its "
                        f"labelled chunks {query['relevant']}"
                    )
        else:
            raise ValueError(f"query {query['id']} has unknown gold_answer kind {kind!r}")

    type_counts: Dict[str, int] = {}
    for doc in documents:
        type_counts[doc["metadata"]["type"]] = type_counts.get(doc["metadata"]["type"], 0) + 1
    category_counts: Dict[str, int] = {}
    for query in queries:
        category_counts[query["category"]] = category_counts.get(query["category"], 0) + 1

    return {
        "documents": len(documents),
        "document_types": type_counts,
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
        "queries": len(queries),
        "query_categories": category_counts,
        "labelled_ids": len({node_id for query in queries for node_id in query["relevant"]}),
        "gold_answer_kinds": gold_kinds,
    }


def main() -> None:
    rng = random.Random(SEED)
    incidents = build_incidents()
    documents = build_documents(incidents, rng)
    graph = build_graph(documents, incidents)
    queries = build_queries(incidents)
    summary = validate(documents, graph, queries)

    (HERE / "corpus_documents.json").write_text(
        json.dumps(documents, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (HERE / "corpus_graph.json").write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (HERE / "dataset.json").write_text(
        json.dumps(
            {
                "corpus_id": CORPUS_ID,
                "dataset_version": DATASET_VERSION,
                "generator": "benchmark_corpora/meridian/generate_corpus.py",
                "generator_seed": SEED,
                "notes": (
                    "Labels are generated from the same entity tables as the text and "
                    "the graph. Multi-hop families are written so the answer chunk has "
                    "no station or line vocabulary, forcing graph traversal."
                ),
                "queries": queries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nWrote corpus_documents.json, corpus_graph.json, dataset.json to {HERE}")


if __name__ == "__main__":
    main()
