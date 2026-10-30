"""Nonparametric ID, scoped probability expressions, and finite-distribution evaluation."""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

from ._serialization import freeze, thaw
from .graph import ADMG


@dataclass(frozen=True)
class Expression:
    """Immutable expression with explicitly bound marginalization variables."""

    op: str
    variables: tuple[str, ...] = ()
    given: tuple[str, ...] = ()
    terms: tuple[Expression, ...] = ()

    @cached_property
    def free(self) -> frozenset[str]:
        """Return all free random and fixed variables in this expression."""
        if self.op == "probability":
            return frozenset(self.variables + self.given)
        result = frozenset().union(*(t.free for t in self.terms))
        return result - set(self.variables) if self.op == "marginal" else result

    def to_dict(self) -> dict:
        """Serialize an expression without executable code."""
        return {
            "op": self.op,
            "variables": list(self.variables),
            "given": list(self.given),
            "terms": [t.to_dict() for t in self.terms],
        }

    def __str__(self) -> str:
        if self.op == "one":
            return "1"
        if self.op == "probability":
            suffix = " | " + ", ".join(self.given) if self.given else ""
            return "P(" + ", ".join(self.variables) + suffix + ")"
        if self.op == "product":
            return " * ".join(f"({t})" for t in self.terms)
        if self.op == "quotient":
            return f"({self.terms[0]}) / ({self.terms[1]})"
        return f"marginalize[{', '.join(self.variables)}]({self.terms[0]})"

    def to_latex(self) -> str:
        """Render notation with escaped node names, using a generic marginal operator."""

        def label(v):
            return r"\mathrm{" + "".join(c if c.isalnum() else r"\_" for c in v) + "}"

        if self.op == "one":
            return "1"
        if self.op == "probability":
            tail = r"\mid " + ",".join(map(label, self.given)) if self.given else ""
            return "P(" + ",".join(map(label, self.variables)) + tail + ")"
        if self.op == "product":
            return " ".join(r"\left(" + t.to_latex() + r"\right)" for t in self.terms)
        if self.op == "quotient":
            return r"\frac{" + self.terms[0].to_latex() + "}{" + self.terms[1].to_latex() + "}"
        return (
            r"\mathop{\mathrm{Marg}}_{"
            + ",".join(map(label, self.variables))
            + "}"
            + self.terms[0].to_latex()
        )

    def evaluate(
        self, distribution: DiscreteDistribution, assignment: Mapping[str, object]
    ) -> float:
        """Evaluate exactly on a supplied finite joint distribution; reject unsupported points."""
        if not self.free <= assignment.keys():
            raise ValueError(f"Missing assignments: {sorted(self.free - assignment.keys())}")
        if self.op == "one":
            return 1.0
        if self.op == "probability":
            numerator = distribution.probability(
                {n: assignment[n] for n in self.variables + self.given}
            )
            denominator = distribution.probability({n: assignment[n] for n in self.given})
            if denominator <= 0:
                raise ValueError("Conditional probability outside observed support")
            return numerator / denominator
        if self.op == "product":
            result = 1.0
            for term in self.terms:
                result *= term.evaluate(distribution, assignment)
                if result == 0:
                    break
            return result
        if self.op == "quotient":
            denominator = self.terms[1].evaluate(distribution, assignment)
            if denominator <= 0:
                raise ValueError("Kernel denominator outside observed support")
            return self.terms[0].evaluate(distribution, assignment) / denominator
        if self.op == "marginal":
            return math.fsum(
                self.terms[0].evaluate(
                    distribution, dict(assignment) | dict(zip(self.variables, values, strict=True))
                )
                for values in itertools.product(*(distribution.domains[n] for n in self.variables))
            )
        raise ValueError("Unknown expression operator")


def probability(variables, given=()) -> Expression:
    """Construct an observational joint or conditional probability."""
    variables, given = tuple(sorted(variables)), tuple(sorted(given))
    return Expression("probability", variables, given) if variables else Expression("one")


def marginal(variables, term: Expression) -> Expression:
    """Bind variables while simplifying observational marginals when valid."""
    variables = frozenset(variables) & term.free
    if not variables:
        return term
    if term.op == "probability" and not variables & set(term.given):
        return probability(set(term.variables) - variables, term.given)
    if term.op == "marginal":
        return marginal(variables | set(term.variables), term.terms[0])
    return Expression("marginal", tuple(sorted(variables)), terms=(term,))


def product(terms) -> Expression:
    """Flatten products and remove identity factors without changing scope."""
    factors = []
    for term in terms:
        factors.extend(
            term.terms if term.op == "product" else (() if term.op == "one" else (term,))
        )
    if not factors:
        return Expression("one")
    if len(factors) == 1:
        return factors[0]
    return Expression("product", terms=tuple(factors))


def chain_factor(p: Expression, order: tuple[str, ...], index: int) -> Expression:
    """Derive a chain-rule factor from a joint kernel without discarding its fixed variables."""
    if p.op == "probability" and set(p.variables) == set(order):
        return probability((order[index],), order[:index] + p.given)
    numerator = marginal(order[index + 1 :], p)
    denominator = marginal(order[index:], p)
    if denominator.op == "one":
        return numerator
    return Expression("quotient", terms=(numerator, denominator))


class DiscreteDistribution:
    """Validated finite observational joint distribution with memoized marginals."""

    def __init__(self, nodes, masses):
        self.nodes = tuple(nodes)
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Duplicate distribution variables")
        self.masses = {tuple(k): float(v) for k, v in masses.items()}
        if any(
            len(k) != len(self.nodes) or not math.isfinite(v) or v < 0
            for k, v in self.masses.items()
        ):
            raise ValueError("Invalid joint probability table")
        if not math.isclose(math.fsum(self.masses.values()), 1.0, abs_tol=1e-10):
            raise ValueError("Probability masses must sum to one")
        self.domains = {
            n: tuple(dict.fromkeys(k[i] for k in self.masses)) for i, n in enumerate(self.nodes)
        }
        self.masses = MappingProxyType(self.masses)
        self.domains = MappingProxyType(self.domains)
        self._cache = {}

    def probability(self, assignment) -> float:
        """Return a marginal probability for a partial assignment."""
        if not set(assignment) <= set(self.nodes):
            raise ValueError("Unknown distribution variable")
        key = tuple(sorted(assignment.items()))
        if key not in self._cache:
            positions = [(self.nodes.index(n), v) for n, v in key]
            self._cache[key] = math.fsum(
                p for row, p in self.masses.items() if all(row[i] == v for i, v in positions)
            )
        return self._cache[key]


@dataclass(frozen=True)
class IdentificationResult:
    """Identification outcome; failure distinguishes mathematical and computational limits."""

    status: str
    expression: Expression | None = None
    trace: tuple[dict, ...] = ()
    witness: dict | None = None
    reason: str = ""
    method: str = "id"
    adjustment: tuple[str, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "trace", tuple(freeze(v) for v in self.trace))
        object.__setattr__(self, "witness", freeze(self.witness))

    def to_dict(self):
        """Serialize the result and its algorithm certificate."""
        return {
            "status": self.status,
            "method": self.method,
            "adjustment": list(self.adjustment) if self.adjustment is not None else None,
            "expression": self.expression.to_dict() if self.expression else None,
            "formula": str(self.expression) if self.expression else None,
            "trace": thaw(self.trace),
            "witness": thaw(self.witness),
            "reason": self.reason,
        }


class _Failure(Exception):
    def __init__(self, graph, district, treatment, outcome):
        self.graph, self.district, self.treatment, self.outcome = (
            graph,
            district,
            treatment,
            outcome,
        )


def _hedge(failure: _Failure) -> dict:
    g, s = failure.graph, failure.district
    roots = {n for n in s if not (g.children[n] & s)}
    chosen = []
    reaching = set(roots)
    for n in reversed(g.topological_order):
        if n in roots:
            continue
        candidates = g.children[n] & reaching
        if n in s:
            candidates &= s
        if candidates:
            chosen.append((n, min(candidates)))
            reaching.add(n)
    if reaching != set(g.nodes):
        raise RuntimeError("Unable to construct the ID failure forest")
    outer = ADMG(g.nodes, tuple(chosen), g.bidirected)
    inner = outer.induced(s)
    certificate = {
        "outer": outer.to_dict(),
        "inner": inner.to_dict(),
        "subproblem": g.to_dict(),
        "treatment": sorted(failure.treatment),
        "outcome": sorted(failure.outcome),
        "roots": sorted(roots),
    }
    if not verify_hedge(certificate):
        raise RuntimeError("Internal hedge verification failed")
    return certificate


def verify_hedge(certificate: dict) -> bool:
    """Independently check the C-forest hedge conditions for the recorded ID subproblem."""
    try:
        outer, inner, graph = (
            ADMG.from_dict(certificate[k]) for k in ("outer", "inner", "subproblem")
        )
        x, y = set(certificate["treatment"]), set(certificate["outcome"])
        roots = set(certificate["roots"])
        graph.validate_nodes(x | y)
        if not roots or not set(inner.nodes) < set(outer.nodes) <= set(graph.nodes):
            return False
        if not set(outer.nodes) & x or set(inner.nodes) & x:
            return False
        if not roots <= graph.intervene(x).ancestors(y):
            return False
        if not set(inner.directed) <= set(outer.directed) <= set(graph.directed):
            return False
        if not set(inner.bidirected) <= set(outer.bidirected) <= set(graph.bidirected):
            return False
        for forest in (inner, outer):
            if len(forest.districts) != 1 or any(len(v) > 1 for v in forest.children.values()):
                return False
            if {n for n in forest.nodes if not forest.children[n]} != roots:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def identify(
    graph: ADMG,
    *,
    treatment,
    outcome,
    method="auto",
    max_sets=10000,
    max_calls: int = 10000,
    timeout: float = 10.0,
) -> IdentificationResult:
    """Apply Shpitser–Pearl ID to a joint intervention under a semi-Markovian ADMG."""
    if type(max_calls) is not int or max_calls < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Identification budgets must be positive and finite")
    x, y = graph.validate_nodes(treatment), graph.validate_nodes(outcome)
    if not x or not y or x & y:
        raise ValueError("Treatment and outcome sets must be nonempty and disjoint")
    if method not in ("auto", "id") or type(max_sets) is not int or max_sets < 1:
        raise ValueError("Use method auto/id and a positive integer max_sets")
    deadline, trace, calls = time.monotonic() + timeout, [], 0
    memo = {}
    if method == "auto" and len(x) == 1:
        try:
            adjustment = _find_backdoor(graph, x, y, max_sets, deadline)
        except TimeoutError:
            adjustment = None
        if adjustment is not None:
            expression = marginal(
                adjustment, product((probability(y, x | set(adjustment)), probability(adjustment)))
            )
            return IdentificationResult(
                "identified",
                expression,
                ({"rule": "backdoor", "adjustment": list(adjustment)},),
                reason="Identified by the sufficient back-door criterion under the supplied causal graph",
                method="backdoor",
                adjustment=adjustment,
            )

    def solve(y, x, p, g):
        nonlocal calls
        calls += 1
        if calls > max_calls or time.monotonic() > deadline:
            raise TimeoutError("Identification budget exhausted")
        key = (y, x, p, g)
        if key in memo:
            return memo[key]
        v = frozenset(g.nodes)

        def record(rule):
            trace.append(
                {"rule": rule, "graph": g.to_dict(), "treatment": sorted(x), "outcome": sorted(y)}
            )

        if not x:
            record("marginalize")
            result = marginal(v - y, p)
        elif (anc := g.ancestors(y)) != v:
            record("restrict_ancestors")
            result = solve(y, x & anc, marginal(v - anc, p), g.induced(anc))
        elif w := (v - x) - g.intervene(x).ancestors(y):
            record("augment_intervention")
            result = solve(y, x | w, p, g)
        elif len(districts := g.induced(v - x).districts) > 1:
            record("factor_districts")
            result = marginal(v - x - y, product(solve(s, v - s, p, g) for s in districts))
        else:
            s = districts[0]
            if len(g.districts) == 1:
                record("hedge_failure")
                raise _Failure(g, s, x, y)
            order = g.topological_order
            if s in g.districts:
                record("district_kernel")
                result = marginal(
                    s - y, product(chain_factor(p, order, i) for i, n in enumerate(order) if n in s)
                )
            else:
                record("restrict_kernel")
                larger = next(d for d in g.districts if s < d)
                kernel = product(
                    chain_factor(p, order, i) for i, n in enumerate(order) if n in larger
                )
                result = solve(y, x & larger, kernel, g.induced(larger))
        memo[key] = result
        return result

    try:
        expression = solve(y, x, probability(graph.nodes), graph)
        extra = expression.free - x - y
        if extra:
            trace.append(
                {"rule": "average_invariant_augmented_assignments", "variables": sorted(extra)}
            )
            expression = marginal(extra, product((expression, probability(extra))))
        return IdentificationResult(
            "identified",
            expression,
            tuple(trace),
            reason="Conditional on the supplied graph and semi-Markovian causal assumptions",
        )
    except _Failure as failure:
        return IdentificationResult(
            "not_identified",
            trace=tuple(trace),
            witness=_hedge(failure),
            reason="A verified hedge witnesses failure in the recorded ID subproblem",
        )
    except (TimeoutError, RecursionError) as exc:
        return IdentificationResult(
            "resource_limited",
            trace=tuple(trace),
            reason=str(exc) or "Python recursion limit reached",
        )


def _find_backdoor(graph, x, y, max_sets, deadline):
    """Find a minimum-cardinality nondescendant set blocking all back-door paths."""
    candidates = sorted(graph.ancestors(x | y) - graph.descendants(x) - y)
    backdoor = type(graph)(
        graph.nodes, tuple((u, v) for u, v in graph.directed if u not in x), graph.bidirected
    )
    checked = 0
    for size in range(len(candidates) + 1):
        for z in itertools.combinations(candidates, size):
            if checked >= max_sets or time.monotonic() > deadline:
                raise TimeoutError("Back-door search budget exhausted")
            checked += 1
            if backdoor.separated(x, y, z):
                return z
    return None
