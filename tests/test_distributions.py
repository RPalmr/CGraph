"""Exact factorization, intervention normalization, and observational/interventional distinctions."""

import itertools

import pytest

from cgraph import ADMG, DAG, DiscreteDistribution, factorize, identify, intervene


def fixture():
    graph = DAG(("z", "t", "y"), (("z", "t"), ("z", "y"), ("t", "y")))
    masses = {}
    for z, t, y in itertools.product((0, 1), repeat=3):
        pt, py = 0.2 + 0.6 * z, 0.1 + 0.3 * z + 0.4 * t
        masses[z, t, y] = 0.5 * (pt if t else 1 - pt) * (py if y else 1 - py)
    return graph, DiscreteDistribution(("z", "t", "y"), masses)


def test_factorization_matches_joint_law():
    graph, observed = fixture()
    formula = factorize(graph)
    for values, mass in observed.masses.items():
        assert formula.evaluate(
            observed, dict(zip(observed.nodes, values, strict=True))
        ) == pytest.approx(mass)
    assert factorize(DAG(())).evaluate(DiscreteDistribution((), {(): 1}), {}) == 1


def test_intervention_is_normalized_and_differs_from_observational_conditioning():
    graph, observed = fixture()
    post = intervene(graph, {"t": 1})
    assert post.graph.directed == (("t", "y"), ("z", "y"))
    assert sum(
        post.evaluate(observed, {"z": z, "y": y}) for z, y in itertools.product((0, 1), repeat=2)
    ) == pytest.approx(1)
    marginal = post.marginalize("y")
    assert marginal.evaluate(observed, {"y": 1}) == pytest.approx(0.65)
    observed_conditional = observed.probability({"y": 1, "t": 1}) / observed.probability({"t": 1})
    assert observed_conditional == pytest.approx(0.74)
    assert marginal.evaluate(observed, {"t": 0, "y": 1}) == 0
    with pytest.raises(TypeError):
        post.values["t"] = 0
    assert intervene(graph, {"t": 1, "z": 0}).evaluate(observed, {"y": 1}) == pytest.approx(0.5)
    assert intervene(graph, {"t": 1, "z": 0, "y": 1}).evaluate(observed, {}) == 1


def test_backdoor_id_and_truncation_agree():
    graph, observed = fixture()
    backdoor = identify(graph, treatment="t", outcome="y")
    general = identify(graph, treatment="t", outcome="y", method="id")
    assert backdoor.method == "backdoor"
    assert backdoor.adjustment == ("z",)
    for t, y in itertools.product((0, 1), repeat=2):
        assignment = {"t": t, "y": y}
        expected = intervene(graph, {"t": t}).marginalize("y").evaluate(observed, {"y": y})
        assert backdoor.expression.evaluate(observed, assignment) == pytest.approx(expected)
        assert general.expression.evaluate(observed, assignment) == pytest.approx(expected)


def test_truncation_rejects_hidden_confounding_and_invalid_values():
    graph, observed = fixture()
    hidden = ADMG(graph.nodes, graph.directed, (("t", "y"),))
    with pytest.raises(ValueError, match="DAG"):
        factorize(hidden)
    with pytest.raises(ValueError, match="DAG"):
        intervene(hidden, {"t": 1})
    for value in ({}, {"unknown": 1}, {"t": float("nan")}, {"t": []}):
        with pytest.raises(ValueError):
            intervene(graph, value)
    with pytest.raises(ValueError, match="domain"):
        intervene(graph, {"t": 7}).marginalize("y").evaluate(observed, {"y": 1})


def test_zero_observational_mass_and_unidentified_intervention_support():
    graph = DAG(("t", "y"), (("t", "y"),))
    observed = DiscreteDistribution(
        ("t", "y"), {(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.0, (1, 1): 0.0}
    )
    assert factorize(graph).evaluate(observed, {"t": 1, "y": 1}) == 0
    with pytest.raises(ValueError, match="support"):
        intervene(graph, {"t": 1}).evaluate(observed, {"y": 1})


def test_multiple_treatment_identification_matches_truncation():
    graph = DAG(("a", "b", "y"), (("a", "b"), ("a", "y"), ("b", "y")))
    masses = {}
    for a, b, y in itertools.product((0, 1), repeat=3):
        pb, py = 0.2 + 0.5 * a, 0.1 + 0.2 * a + 0.4 * b
        masses[a, b, y] = 0.5 * (pb if b else 1 - pb) * (py if y else 1 - py)
    observed = DiscreteDistribution(("a", "b", "y"), masses)
    result = identify(graph, treatment=("a", "b"), outcome="y")
    assert result.status == "identified"
    assert result.method == "id"
    for a, b in itertools.product((0, 1), repeat=2):
        expected = intervene(graph, {"a": a, "b": b}).evaluate(observed, {"y": 1})
        assert result.expression.evaluate(observed, {"a": a, "b": b, "y": 1}) == pytest.approx(
            expected
        )
