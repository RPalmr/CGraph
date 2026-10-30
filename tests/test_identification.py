"""Exact finite-SCM and nonidentification certificate tests."""

import itertools

import pytest

from cgraph import ADMG, DAG, CausalQuery
from cgraph.symbolic import DiscreteDistribution, identify, identify_family, verify_hedge


def test_backdoor_bow_and_budget():
    query = CausalQuery.interventional(treatment="t", outcome="y")
    dag = DAG(("z", "t", "y"), (("z", "t"), ("z", "y"), ("t", "y")))
    result = identify(dag, query)
    assert result.status == "identified"
    assert result.expression.free == {"t", "y"}
    bow = ADMG(("t", "y"), (("t", "y"),), (("t", "y"),))
    failure = identify(bow, query)
    assert failure.status == "not_identified"
    assert verify_hedge(failure.witness)
    assert identify(dag, query, max_calls=1).status == "resource_limited"
    family = identify_family([dag, bow], query)
    assert not family["common_functional_established"]


def test_frontdoor_against_explicit_latent_intervention():
    graph = ADMG(("t", "m", "y"), (("t", "m"), ("m", "y")), (("t", "y"),))
    result = identify(graph, CausalQuery.interventional(treatment="t", outcome="y"))
    assert result.status == "identified"
    masses = {}
    for u, t, m, y in itertools.product((0, 1), repeat=4):
        pt, pm, py = 0.2 + 0.6 * u, 0.25 + 0.5 * t, 0.1 + 0.3 * u + 0.4 * m
        p = 0.5 * (pt if t else 1 - pt) * (pm if m else 1 - pm) * (py if y else 1 - py)
        masses[t, m, y] = masses.get((t, m, y), 0) + p
    joint = DiscreteDistribution(("t", "m", "y"), masses)
    for t in (0, 1):
        expected = sum(
            0.5 * ((0.25 + 0.5 * t) if m else (0.75 - 0.5 * t)) * (0.1 + 0.3 * u + 0.4 * m)
            for u, m in itertools.product((0, 1), repeat=2)
        )
        assert result.expression.evaluate(joint, {"t": t, "y": 1}) == pytest.approx(expected)


def test_all_three_node_admgs_have_valid_results_and_certificates():
    nodes = tuple("abc")
    edges = tuple(itertools.combinations(nodes, 2))
    for directed_mask, bidirected_mask in itertools.product(range(8), repeat=2):
        graph = ADMG(
            nodes,
            tuple(e for i, e in enumerate(edges) if directed_mask & (1 << i)),
            tuple(e for i, e in enumerate(edges) if bidirected_mask & (1 << i)),
        )
        for t, y in itertools.permutations(nodes, 2):
            result = identify(graph, CausalQuery.interventional(treatment=t, outcome=y))
            assert result.status in ("identified", "not_identified")
            if result.witness:
                assert verify_hedge(result.witness)
            else:
                assert result.expression.free <= {t, y}


def test_random_admgs_against_exact_latent_scm_interventions():
    import math
    import random

    rng = random.Random(1907)
    nodes = tuple("abcd")
    edges = tuple(itertools.combinations(nodes, 2))
    for _ in range(24):
        graph = ADMG(
            nodes,
            tuple(e for e in edges if rng.random() < 0.45),
            tuple(e for e in edges if rng.random() < 0.3),
        )
        coefficients = {n: rng.uniform(-0.7, 0.7) for n in nodes}

        def joint(intervention=None, graph=graph, coefficients=coefficients):
            result = {}
            for hidden in itertools.product((0, 1), repeat=len(graph.bidirected)):
                for observed in itertools.product((0, 1), repeat=4):
                    assignment = dict(zip(nodes, observed, strict=True))
                    mass = 0.5 ** len(hidden)
                    for n in nodes:
                        if intervention and n == intervention[0]:
                            mass *= assignment[n] == intervention[1]
                            continue
                        logit = (
                            coefficients[n]
                            + 0.4 * sum(assignment[p] for p in graph.parents[n])
                            + 0.6
                            * sum(hidden[i] for i, edge in enumerate(graph.bidirected) if n in edge)
                        )
                        p = 1 / (1 + math.exp(-logit))
                        mass *= p if assignment[n] else 1 - p
                    result[observed] = result.get(observed, 0) + mass
            return DiscreteDistribution(nodes, result)

        observational = joint()
        for t, y in itertools.permutations(nodes, 2):
            result = identify(graph, CausalQuery.interventional(treatment=t, outcome=y))
            if result.status == "identified":
                actual = result.expression.evaluate(observational, {t: 1, y: 1})
                expected = joint((t, 1)).probability({y: 1})
                assert actual == pytest.approx(expected, abs=1e-10), (graph, t, y, result)
            from cgraph.symbolic import identify_conditional

            conditioned = next(v for v in nodes if v not in (t, y))
            conditional = identify_conditional(
                graph, CausalQuery.interventional(treatment=t, outcome=y), given=(conditioned,)
            )
            if conditional.status == "identified":
                intervened = joint((t, 1))
                expected_conditional = intervened.probability(
                    {y: 1, conditioned: 1}
                ) / intervened.probability({conditioned: 1})
                actual_conditional = conditional.expression.evaluate(
                    observational, {t: 1, y: 1, conditioned: 1}
                )
                assert actual_conditional == pytest.approx(expected_conditional, abs=1e-10)


def test_conditional_identification_uses_exchange_before_joint_identification():
    from cgraph.symbolic import identify_conditional

    graph = ADMG(("x", "z", "y"), (("x", "z"), ("z", "y")), (("x", "z"),))
    query = CausalQuery.interventional(treatment="x", outcome="y")
    assert identify(graph, query).status == "not_identified"
    result = identify_conditional(graph, query, given=("z",))
    assert result.status == "identified"
    assert result.trace[0]["rule"] == "idc_exchange_observation"
    with pytest.raises(TypeError):
        result.trace[0]["rule"] = "modified"
    with pytest.raises(ValueError):
        identify_conditional(graph, query, given=("x",))


def test_structural_scenarios_and_joint_intervention():
    from cgraph.scenarios import add_latent_confounding, audit_graph_scenarios

    graph = DAG(("a", "b", "y"), (("a", "y"), ("b", "y")))
    query = CausalQuery.interventional(treatment=("a", "b"), outcome="y")
    assert identify(graph, query).status == "identified"
    scenario = add_latent_confounding(graph, (("a", "y"),))
    check = audit_graph_scenarios(graph, query, [("hidden_common_cause", scenario)])
    assert check.metrics["scenarios"][0]["identification"]["status"] == "not_identified"
