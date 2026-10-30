"""Markov factorization and truncated-factorization intervention distributions."""

from collections.abc import Mapping
from dataclasses import dataclass

from ._serialization import freeze, thaw
from .graph import DAG
from .symbolic import Expression, marginal, probability, product


def factorize(graph: DAG) -> Expression:
    """Return product_i P(V_i | parents_i) under the supplied fully observed causal DAG.

    This constructs the assumed Markov factorization. It does not establish
    graph correctness or test whether a supplied observational law is Markov.
    """
    if not isinstance(graph, DAG):
        raise ValueError("Observed-parent factorization requires a fully observed DAG")
    return product(probability((n,), graph.parents[n]) for n in graph.topological_order)


@dataclass(frozen=True)
class InterventionDistribution:
    """Truncated distribution over nonintervened variables with fixed treatment values."""

    expression: Expression
    graph: DAG
    values: Mapping
    variables: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "values", freeze(self.values))

    def evaluate(self, distribution, assignment) -> float:
        """Evaluate the interventional law, returning zero for incompatible fixed assignments."""
        if not set(self.variables) <= assignment.keys():
            raise ValueError("Missing nonintervened outcome assignments")
        for name, value in self.values.items():
            if name not in distribution.domains or value not in distribution.domains[name]:
                raise ValueError("Intervention value lies outside the observed variable domain")
            if name in assignment and assignment[name] != value:
                return 0.0
        return self.expression.evaluate(distribution, dict(assignment) | dict(self.values))

    def marginalize(self, keep):
        """Marginalize other random variables while retaining the fixed intervention."""
        keep = (keep,) if isinstance(keep, str) else tuple(keep)
        if len(set(keep)) != len(keep) or not set(keep) <= set(self.variables):
            raise ValueError("Keep only distinct nonintervened random variables")
        return InterventionDistribution(
            marginal(set(self.variables) - set(keep), self.expression),
            self.graph,
            self.values,
            tuple(sorted(keep)),
        )

    def to_dict(self):
        """Serialize the truncated expression, fixed assignments, and mutilated graph."""
        return {
            "expression": self.expression.to_dict(),
            "formula": str(self.expression),
            "interventions": thaw(self.values),
            "variables": list(self.variables),
            "mutilated_graph": self.graph.to_dict(),
        }


def intervene(graph: DAG, values: Mapping) -> InterventionDistribution:
    """Drop treatment mechanisms and fix values in the remaining original-parent factors.

    The returned law is on V minus the intervention nodes. Its extension to the
    full joint law has point mass at each specified treatment value. Continuous
    interventions are interpreted as fixed parameters rather than probability
    masses. Numerical evaluation is currently for finite discrete laws only.
    """
    if not isinstance(graph, DAG):
        raise ValueError("Truncated observational factorization requires a fully observed DAG")
    if not isinstance(values, Mapping) or not values:
        raise ValueError("Supply a nonempty intervention mapping")
    targets = graph.validate_nodes(values)
    values = freeze(values)
    if any(
        value is None or not isinstance(value, (str, bool, int, float)) for value in values.values()
    ):
        raise ValueError("Intervention values must be finite scalar states")
    expression = product(
        probability((n,), graph.parents[n]) for n in graph.topological_order if n not in targets
    )
    return InterventionDistribution(
        expression,
        graph.intervene(targets),
        values,
        tuple(n for n in graph.nodes if n not in targets),
    )
