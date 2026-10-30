"""Structural sensitivity scenarios without probabilistic claims about graph alternatives."""

from __future__ import annotations

from .graph import ADMG
from .reports import CheckResult
from .symbolic import identify


def add_latent_confounding(graph, pairs):
    """Create an ADMG scenario with additional explicitly specified bidirected edges."""
    return ADMG(graph.nodes, graph.directed, graph.bidirected + tuple(pairs))


def audit_graph_scenarios(graph, query, scenarios, *, max_scenarios=100, **identification_options):
    """Compare named candidate graphs with a baseline under a bounded explicit scenario set."""
    if max_scenarios < 1:
        raise ValueError("Scenario budget must be positive")
    baseline = identify(graph, query, **identification_options)
    results, names, truncated = [], set(), False
    for name, candidate in scenarios:
        if len(results) >= max_scenarios:
            truncated = True
            break
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Scenario names must be unique nonempty strings")
        if candidate.nodes != graph.nodes:
            raise ValueError("Structural scenarios must retain the observed variable set")
        names.add(name)
        result = identify(candidate, query, **identification_options)
        results.append(
            {
                "name": name,
                "graph": candidate.to_dict(),
                "identification": result.to_dict(),
                "status_changed": result.status != baseline.status,
            }
        )
    return CheckResult(
        "graph_scenarios",
        "resource_limited" if truncated else "completed",
        "stress_scenario",
        "Structural sensitivity within the supplied candidate scenarios; no probabilities assigned",
        None if truncated else "informational",
        {"baseline": baseline.to_dict(), "scenarios": results, "truncated": truncated},
    )
