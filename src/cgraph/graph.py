"""Immutable causal graphs and exact graph-theoretic operations."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType


def _names(values: Iterable[str]) -> frozenset[str]:
    """Interpret a single name as one variable, never as individual characters."""
    return frozenset((values,)) if isinstance(values, str) else frozenset(values)


@dataclass(frozen=True)
class ADMG:
    """Acyclic directed mixed graph; directed/bidirected bows are permitted."""

    nodes: tuple[str, ...]
    directed: tuple[tuple[str, str], ...] = ()
    bidirected: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        if any(not isinstance(n, str) or not n for n in nodes):
            raise ValueError("Node identifiers must be nonempty strings")
        if len(nodes) != len(set(nodes)):
            raise ValueError("Duplicate node identifiers")
        edges = []
        for collection, symmetric in ((self.directed, False), (self.bidirected, True)):
            clean = set()
            for edge in collection:
                if len(edge) != 2:
                    raise ValueError("Edges must have two endpoints")
                u, v = edge
                if u == v or u not in nodes or v not in nodes:
                    raise ValueError(f"Invalid edge: {edge!r}")
                clean.add(tuple(sorted((u, v))) if symmetric else (u, v))
            edges.append(tuple(sorted(clean)))
        object.__setattr__(self, "nodes", tuple(sorted(nodes)))
        object.__setattr__(self, "directed", edges[0])
        object.__setattr__(self, "bidirected", edges[1])
        if isinstance(self, DAG) and self.bidirected:
            raise ValueError("DAG cannot contain bidirected edges")
        _ = self.topological_order

    @cached_property
    def parents(self) -> dict[str, frozenset[str]]:
        """Return immutable parent sets indexed by node."""
        result = {n: set() for n in self.nodes}
        for u, v in self.directed:
            result[v].add(u)
        return MappingProxyType({n: frozenset(v) for n, v in result.items()})

    @cached_property
    def children(self) -> dict[str, frozenset[str]]:
        """Return immutable child sets indexed by node."""
        result = {n: set() for n in self.nodes}
        for u, v in self.directed:
            result[u].add(v)
        return MappingProxyType({n: frozenset(v) for n, v in result.items()})

    @cached_property
    def topological_order(self) -> tuple[str, ...]:
        """Return a stable topological ordering, rejecting directed cycles."""
        degree = {n: len(self.parents[n]) for n in self.nodes}
        queue = deque(n for n in self.nodes if degree[n] == 0)
        result = []
        while queue:
            n = queue.popleft()
            result.append(n)
            for child in sorted(self.children[n]):
                degree[child] -= 1
                if degree[child] == 0:
                    queue.append(child)
        if len(result) != len(self.nodes):
            raise ValueError("Directed cycle detected")
        return tuple(result)

    def validate_nodes(self, nodes: Iterable[str]) -> frozenset[str]:
        """Validate and normalize a set of node names."""
        result = _names(nodes)
        if not result <= set(self.nodes):
            raise ValueError(f"Unknown nodes: {sorted(result - set(self.nodes))}")
        return result

    def _closure(self, nodes: Iterable[str], adjacency) -> frozenset[str]:
        seen = set(self.validate_nodes(nodes))
        stack = list(seen)
        while stack:
            for adjacent in adjacency[stack.pop()]:
                if adjacent not in seen:
                    seen.add(adjacent)
                    stack.append(adjacent)
        return frozenset(seen)

    def ancestors(self, nodes: Iterable[str]) -> frozenset[str]:
        """Return directed ancestors including the supplied nodes."""
        return self._closure(nodes, self.parents)

    def descendants(self, nodes: Iterable[str]) -> frozenset[str]:
        """Return directed descendants including the supplied nodes."""
        return self._closure(nodes, self.children)

    def induced(self, nodes: Iterable[str]) -> ADMG:
        """Return the induced subgraph, preserving its graph class."""
        keep = self.validate_nodes(nodes)
        return type(self)(
            tuple(keep),
            tuple((u, v) for u, v in self.directed if u in keep and v in keep),
            tuple((u, v) for u, v in self.bidirected if u in keep and v in keep),
        )

    def intervene(self, nodes: Iterable[str]) -> ADMG:
        """Remove incoming arrowheads, including incident bidirected edges."""
        targets = self.validate_nodes(nodes)
        return type(self)(
            self.nodes,
            tuple((u, v) for u, v in self.directed if v not in targets),
            tuple((u, v) for u, v in self.bidirected if u not in targets and v not in targets),
        )

    @cached_property
    def districts(self) -> tuple[frozenset[str], ...]:
        """Return bidirected connected components including singleton nodes."""
        adjacency = {n: set() for n in self.nodes}
        for u, v in self.bidirected:
            adjacency[u].add(v)
            adjacency[v].add(u)
        unseen = set(self.nodes)
        result = []
        while unseen:
            component = self._closure((min(unseen),), adjacency)
            unseen.difference_update(component)
            result.append(component)
        return tuple(result)

    def separated(self, x: Iterable[str], y: Iterable[str], given: Iterable[str] = ()) -> bool:
        """Test m-separation via an exact latent expansion and ancestral moralization."""
        x, y, z = (self.validate_nodes(v) for v in (x, y, given))
        if x & y or (x | y) & z:
            raise ValueError("Separation arguments must be disjoint")
        parents = {n: set(p) for n, p in self.parents.items()}
        for i, (u, v) in enumerate(self.bidirected):
            latent = ("latent", i)
            parents[latent] = set()
            parents[u].add(latent)
            parents[v].add(latent)
        ancestral = set(x | y | z)
        stack = list(ancestral)
        while stack:
            for p in parents[stack.pop()]:
                if p not in ancestral:
                    ancestral.add(p)
                    stack.append(p)
        moral = {n: set() for n in ancestral}
        for n in ancestral:
            ps = parents[n] & ancestral
            for p in ps:
                moral[n].add(p)
                moral[p].add(n)
            for a, b in itertools.combinations(ps, 2):
                moral[a].add(b)
                moral[b].add(a)
        seen = set(x)
        stack = list(x)
        while stack:
            n = stack.pop()
            if n in y:
                return False
            for adjacent in moral[n] - z - seen:
                seen.add(adjacent)
                stack.append(adjacent)
        return True

    def to_dict(self) -> dict:
        """Serialize the graph without cached implementation details."""
        return {
            "kind": "dag" if isinstance(self, DAG) else "admg",
            "nodes": list(self.nodes),
            "directed": [list(e) for e in self.directed],
            "bidirected": [list(e) for e in self.bidirected],
        }

    @classmethod
    def from_dict(cls, value: dict) -> ADMG:
        """Parse the explicit graph schema; reject unknown graph kinds and keys."""
        if not isinstance(value, dict) or set(value) - {"kind", "nodes", "directed", "bidirected"}:
            raise ValueError("Invalid graph schema")
        kind = value.get("kind")
        if kind not in ("dag", "admg"):
            raise ValueError("Graph kind must be dag or admg")
        return (DAG if kind == "dag" else ADMG)(
            tuple(value["nodes"]),
            tuple(value.get("directed", ())),
            tuple(value.get("bidirected", ())),
        )

    @cached_property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of graph semantics."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class DAG(ADMG):
    """Observed-variable causal DAG without bidirected edges."""


@dataclass(frozen=True)
class AdjustmentResult:
    """Outcome of the complete DAG adjustment criterion for total effects."""

    valid: bool
    reason: str
    forbidden: tuple[str, ...] = ()


def validate_adjustment(graph: DAG, treatment, outcome, adjustment) -> AdjustmentResult:
    """Apply the forbidden-set and proper-backdoor criterion for DAG total effects."""
    if not isinstance(graph, DAG):
        raise ValueError("Adjustment validation currently requires a DAG")
    x, y, z = (graph.validate_nodes(v) for v in (treatment, outcome, adjustment))
    if not x or not y or x & y or (x | y) & z:
        raise ValueError("Treatment, outcome, and adjustment must be disjoint; targets nonempty")
    without_incoming = graph.intervene(x)
    without_outgoing = DAG(graph.nodes, tuple((u, v) for u, v in graph.directed if u not in x))
    causal = (without_incoming.descendants(x) - x) & without_outgoing.ancestors(y)
    forbidden = without_incoming.descendants(causal)
    if z & forbidden:
        return AdjustmentResult(
            False,
            "Adjustment contains a forbidden causal-path descendant",
            tuple(sorted(z & forbidden)),
        )
    backdoor = DAG(
        graph.nodes, tuple((u, v) for u, v in graph.directed if not (u in x and v in causal))
    )
    if not backdoor.separated(x, y, z):
        return AdjustmentResult(False, "Adjustment leaves an open noncausal path")
    return AdjustmentResult(True, "Adjustment satisfies the DAG adjustment criterion")


def find_adjustment(
    graph: DAG, treatment, outcome, *, max_sets: int = 10000
) -> tuple[str, ...] | None:
    """Find a minimum-cardinality set; raise on exhaustion of the computational budget."""
    x, y = graph.validate_nodes(treatment), graph.validate_nodes(outcome)
    if max_sets < 1:
        raise ValueError("max_sets must be positive")
    candidates = sorted(graph.ancestors(x | y) - x - y)
    checked = 0
    for size in range(len(candidates) + 1):
        for z in itertools.combinations(candidates, size):
            if checked >= max_sets:
                raise TimeoutError("Adjustment search budget exhausted")
            checked += 1
            if validate_adjustment(graph, x, y, z).valid:
                return z
    return None


def latent_projection(graph: DAG, observed: Iterable[str]) -> ADMG:
    """Project a DAG by tracing latent-only directed paths and latent common ancestors."""
    if not isinstance(graph, DAG):
        raise ValueError("Projection requires an explicit DAG")
    observed = graph.validate_nodes(observed)
    latent = set(graph.nodes) - observed

    def frontier(start):
        reached, seen, stack = set(), set(), list(graph.children[start])
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in observed:
                reached.add(n)
            else:
                stack.extend(graph.children[n])
        return reached

    directed = {(u, v) for u in observed for v in frontier(u)}
    bidirected = {
        tuple(sorted(pair)) for u in latent for pair in itertools.combinations(frontier(u), 2)
    }
    return ADMG(tuple(observed), tuple(directed), tuple(bidirected))
