"""Immutable causal graphs and exact graph-theoretic operations."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Mapping
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
        node_set = set(nodes)
        if len(nodes) != len(node_set):
            raise ValueError("Duplicate node identifiers")
        edges = []
        for collection, symmetric in ((self.directed, False), (self.bidirected, True)):
            clean = set()
            for edge in collection:
                if isinstance(edge, str) or not isinstance(edge, (tuple, list)) or len(edge) != 2:
                    raise ValueError("Edges must have two endpoints")
                u, v = edge
                if u == v or u not in node_set or v not in node_set:
                    raise ValueError(f"Invalid edge: {edge!r}")
                clean.add(tuple(sorted((u, v))) if symmetric else (u, v))
            edges.append(tuple(sorted(clean)))
        object.__setattr__(self, "nodes", tuple(sorted(nodes)))
        object.__setattr__(self, "directed", edges[0])
        object.__setattr__(self, "bidirected", edges[1])
        if isinstance(self, DAG) and self.bidirected:
            raise ValueError("DAG cannot contain bidirected edges")
        _ = self.topological_order

    def __deepcopy__(self, memo):
        """Share immutable graph state safely when cloning estimator parameters."""
        return self

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
        """Test m-separation in linear traversal time using a latent-expanded active-trail search."""
        x, y, z = (self.validate_nodes(v) for v in (x, y, given))
        if x & y or (x | y) & z:
            raise ValueError("Separation arguments must be disjoint")
        parents = {n: set(p) for n, p in self.parents.items()}
        children = {n: set(c) for n, c in self.children.items()}
        for i, (u, v) in enumerate(self.bidirected):
            latent = ("latent", i)
            parents[latent] = set()
            children[latent] = {u, v}
            parents[u].add(latent)
            parents[v].add(latent)
        ancestors = set(z)
        stack = list(z)
        while stack:
            for parent in parents[stack.pop()]:
                if parent not in ancestors:
                    ancestors.add(parent)
                    stack.append(parent)
        queue, visited = [(n, True) for n in x], set()
        while queue:
            node, upward = queue.pop()
            if (node, upward) in visited:
                continue
            visited.add((node, upward))
            if node not in z and node in y:
                return False
            if upward:
                if node not in z:
                    queue.extend((p, True) for p in parents[node])
                    queue.extend((c, False) for c in children[node])
            else:
                if node not in z:
                    queue.extend((c, False) for c in children[node])
                if node in ancestors:
                    queue.extend((p, True) for p in parents[node])
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
        if not isinstance(value, Mapping) or set(value) - {
            "kind",
            "nodes",
            "directed",
            "bidirected",
        }:
            raise ValueError("Invalid graph schema")
        if "nodes" not in value or not isinstance(value["nodes"], (tuple, list)):
            raise ValueError("Graph nodes must be an explicit sequence")
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
