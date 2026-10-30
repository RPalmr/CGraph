"""Independent graph oracles and structural invariants."""

import itertools

import networkx as nx
import pytest

from cgraph import ADMG, DAG, find_adjustment, latent_projection, validate_adjustment


def test_graph_validation_and_interventions():
    with pytest.raises(ValueError, match="cycle"):
        DAG(("a", "b"), (("a", "b"), ("b", "a")))
    with pytest.raises(ValueError):
        DAG(("a", "b"), (), (("a", "b"),))
    graph = ADMG(("a", "b"), (("a", "b"),), (("a", "b"),))
    assert graph.intervene("b").directed == ()
    assert graph.intervene("b").bidirected == ()
    assert ADMG.from_dict(graph.to_dict()) == graph
    assert graph.districts == (frozenset(("a", "b")),)


def test_collider_descendant():
    graph = DAG(("a", "b", "c", "d"), (("a", "c"), ("b", "c"), ("c", "d")))
    assert graph.separated("a", "b")
    assert not graph.separated("a", "b", "c")
    assert not graph.separated("a", "b", "d")


def test_exhaustive_four_node_dags_against_networkx():
    nodes = tuple("abcd")
    edges = tuple(itertools.combinations(nodes, 2))
    for mask in range(1 << len(edges)):
        selected = tuple(e for i, e in enumerate(edges) if mask & (1 << i))
        graph = DAG(nodes, selected)
        oracle = nx.DiGraph()
        oracle.add_nodes_from(nodes)
        oracle.add_edges_from(selected)
        for a, b in itertools.combinations(nodes, 2):
            remaining = sorted(set(nodes) - {a, b})
            for flags in itertools.product((False, True), repeat=len(remaining)):
                z = {v for v, keep in zip(remaining, flags, strict=True) if keep}
                assert graph.separated(a, b, z) == nx.is_d_separator(oracle, {a}, {b}, z)


def test_adjustment_and_projection():
    graph = DAG(("z", "t", "y"), (("z", "t"), ("z", "y"), ("t", "y")))
    assert not validate_adjustment(graph, "t", "y", ()).valid
    assert find_adjustment(graph, "t", "y") == ("z",)
    mediation = DAG(("t", "m", "y"), (("t", "m"), ("m", "y")))
    assert not validate_adjustment(mediation, "t", "y", ("m",)).valid
    latent = DAG(("u", "t", "y"), (("u", "t"), ("u", "y"), ("t", "y")))
    projected = latent_projection(latent, ("t", "y"))
    assert projected.bidirected == (("t", "y"),)
    assert projected.directed == (("t", "y"),)
    with pytest.raises(TimeoutError):
        find_adjustment(graph, "t", "y", max_sets=1)


def test_admg_separation_against_explicit_latent_dag_oracle():
    nodes = tuple("abc")
    edges = tuple(itertools.combinations(nodes, 2))
    for directed_mask, bidirected_mask in itertools.product(range(8), repeat=2):
        graph = ADMG(
            nodes,
            tuple(e for i, e in enumerate(edges) if directed_mask & (1 << i)),
            tuple(e for i, e in enumerate(edges) if bidirected_mask & (1 << i)),
        )
        oracle = nx.DiGraph()
        oracle.add_nodes_from(nodes)
        oracle.add_edges_from(graph.directed)
        for i, (u, v) in enumerate(graph.bidirected):
            oracle.add_edges_from(((f"latent_{i}", u), (f"latent_{i}", v)))
        for a, b in itertools.combinations(nodes, 2):
            other = set(nodes) - {a, b}
            for given in (set(), other):
                assert graph.separated(a, b, given) == nx.is_d_separator(oracle, {a}, {b}, given)


def test_graph_caches_cannot_be_mutated():
    graph = DAG(("a", "b"), (("a", "b"),))
    with pytest.raises(TypeError):
        graph.parents["b"] = frozenset()
    with pytest.raises(TypeError):
        graph.children["a"] = frozenset()


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "dag", "nodes": "ab"},
        {"kind": "pag", "nodes": ["a", "b"]},
        {"kind": "dag", "nodes": ["a", "b"], "directed": ["ab"]},
        {"kind": "dag"},
    ],
)
def test_malformed_graph_schemas_fail_explicitly(value):
    with pytest.raises(ValueError):
        ADMG.from_dict(value)
