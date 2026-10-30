"""Known causal structures, unresolved equivalence classes, and honest discovery failures."""

import json

import numpy as np
import pytest

from cgraph import DiscoveryResult, PartialEdge, PartialGraph, discover, factorize, identify


def collider(seed=42, n=5000):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(2, n))
    c = a + b + rng.normal(size=n)
    return np.column_stack((a, b, c))


@pytest.fixture(scope="module")
def discovered_collider():
    return discover(collider(), names=("A", "B", "C"), timeout=60)


def test_pc_recovers_collider_and_connects_to_symbolic_core(discovered_collider):
    result = discovered_collider
    assert result.status == "completed", result.to_json()
    assert result.graph.kind == "cpdag"
    assert set(result.graph.directed) == {("A", "C"), ("B", "C")}
    dag = result.graph.to_dag()
    assert identify(dag, treatment="A", outcome="C").status == "identified"
    assert "P(C | A, B)" in str(factorize(dag))
    assert result.diagnostics["unique_ci_tests"] == len(result.tests)
    assert all(0 <= test["p_value"] <= 1 for test in result.tests)
    with pytest.raises(TypeError):
        result.tests[0]["p_value"] = 0


def test_pc_preserves_two_variable_orientation_ambiguity():
    rng = np.random.default_rng(81)
    a = rng.normal(size=2000)
    b = a + rng.normal(size=2000)
    result = discover(np.column_stack((a, b)), names=("A", "B"))
    assert result.status == "completed", result.to_json()
    assert str(result.graph.edges[0]) == "A --- B"
    assert len(result.graph.unresolved) == 1
    with pytest.raises(ValueError, match="unique DAG"):
        result.graph.to_dag()
    with pytest.raises(ValueError, match="unresolved"):
        identify(result.graph, treatment="A", outcome="B")


def test_fci_returns_pag_and_keeps_endpoint_uncertainty():
    result = discover(collider(), names=("A", "B", "C"), method="fci", timeout=60)
    assert result.status == "completed", result.to_json()
    assert result.graph.kind == "pag"
    assert {(e.left, e.right, e.left_mark, e.right_mark) for e in result.graph.edges} == {
        ("A", "C", "circle", "arrow"),
        ("B", "C", "circle", "arrow"),
    }
    with pytest.raises(ValueError):
        result.graph.to_dag()
    assert "Latent common causes allowed" in result.assumptions[-1]


def test_column_order_does_not_choose_directions(discovered_collider):
    data = collider()
    result = discover(data[:, [2, 0, 1]], names=("C", "A", "B"))
    assert result.graph.to_dict() == discovered_collider.graph.to_dict()
    assert result.tests == discovered_collider.tests


def test_budget_and_timeout_do_not_produce_a_complete_graph():
    result = discover(collider(), names=("A", "B", "C"), max_tests=1)
    assert result.status == "resource_limited", result.to_json()
    assert result.graph is None
    assert len(result.tests) == 1
    result = discover(collider(), timeout=0.001)
    assert result.status == "resource_limited"
    assert result.graph is None


def test_required_singular_ci_test_yields_insufficient_data():
    a = np.arange(30, dtype=float)
    result = discover(np.column_stack((a, a)))
    assert result.status == "insufficient_data", result.to_json()
    assert result.graph is None


@pytest.mark.parametrize(
    "data,options",
    [
        (np.ones((30, 2)), {}),
        ([[0, 1], [1, 0]] * 10, {}),
        ([[float("nan"), 2]] * 10, {}),
        (collider(), {"names": ("A", "A", "C")}),
        (collider(), {"method": "unknown"}),
        (collider(), {"alpha": 1}),
        (collider(), {"max_conditioning": -1}),
        (collider(), {"max_tests": 0}),
        (collider(), {"names": ("A", "B")}),
    ],
)
def test_invalid_inputs_are_rejected(data, options):
    with pytest.raises(ValueError):
        discover(data, **options)


def test_json_roundtrip_and_endpoint_dot_export(discovered_collider, tmp_path):
    report = discovered_collider
    restored = DiscoveryResult.from_json(report.to_json(tmp_path / "discovery.json"))
    assert restored.to_dict() == report.to_dict()
    dot = restored.graph.to_dot()
    assert '"A" -> "C"' in dot and "arrowhead=normal" in dot
    with pytest.raises(ValueError):
        DiscoveryResult.from_json(
            report.to_json().replace('"schema_version": "1"', '"schema_version": "2"')
        )
    with pytest.raises(ValueError):
        DiscoveryResult.from_json('{"status":NaN}')
    forged = json.loads(report.to_json())
    forged["graph"]["edges"][0]["left_mark"] = "invented"
    with pytest.raises(ValueError):
        DiscoveryResult.from_json(json.dumps(forged))


def test_pag_and_cpdag_marks_have_distinct_semantics():
    graph = PartialGraph(("A", "B"), (PartialEdge("A", "B", "arrow", "arrow"),), "pag")
    assert not graph.unresolved
    with pytest.raises(ValueError):
        graph.to_dag()
    with pytest.raises(ValueError):
        PartialGraph(graph.nodes, graph.edges, "cpdag")


def test_fci_recovers_latent_confounding_marks_in_identifiable_example():
    rng = np.random.default_rng(117)
    a, b, latent = rng.normal(size=(3, 8000))
    x = a + latent + rng.normal(size=8000)
    y = b + latent + rng.normal(size=8000)
    result = discover(
        np.column_stack((a, b, x, y)), names=("A", "B", "X", "Y"), method="fci", timeout=60
    )
    assert result.status == "completed", result.to_json()
    assert PartialEdge("X", "Y", "arrow", "arrow") in result.graph.edges
    assert result.graph.kind == "pag"


def test_conditioning_budget_and_parent_random_state():
    rng = np.random.default_rng(50)
    a = rng.normal(size=3000)
    b = a + rng.normal(size=3000)
    c = b + rng.normal(size=3000)
    np.random.seed(74)
    before = np.random.get_state()
    result = discover(np.column_stack((a, b, c)), max_conditioning=0)
    after = np.random.get_state()
    assert result.status == "resource_limited", result.to_json()
    assert result.graph is None
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
