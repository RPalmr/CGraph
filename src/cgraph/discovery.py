"""Discover partial causal graphs from observations while preserving orientation uncertainty."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache

from ._serialization import atomic_write, freeze, thaw
from .graph import DAG


@dataclass(frozen=True, order=True)
class PartialEdge:
    """One edge with explicit endpoint marks: tail, arrow, or circle."""

    left: str
    right: str
    left_mark: str
    right_mark: str

    def __post_init__(self):
        if (
            not isinstance(self.left, str)
            or not isinstance(self.right, str)
            or not self.left
            or not self.right
            or self.left >= self.right
            or self.left_mark not in ("tail", "arrow", "circle")
            or self.right_mark not in ("tail", "arrow", "circle")
        ):
            raise ValueError("Edges require canonical distinct endpoints and valid marks")

    def __str__(self):
        left = {"tail": "-", "arrow": "<", "circle": "o"}[self.left_mark]
        right = {"tail": "-", "arrow": ">", "circle": "o"}[self.right_mark]
        return f"{self.left} {left}-{right} {self.right}"

    def to_dict(self):
        """Serialize endpoints without converting ambiguous marks into causal arrows."""
        return {
            "left": self.left,
            "right": self.right,
            "left_mark": self.left_mark,
            "right_mark": self.right_mark,
        }


@dataclass(frozen=True)
class PartialGraph:
    """Estimated CPDAG, PAG, or inconsistent PDAG; circles denote unresolved endpoint marks."""

    nodes: tuple[str, ...]
    edges: tuple[PartialEdge, ...]
    kind: str

    def __post_init__(self):
        nodes = tuple(self.nodes)
        if (
            self.kind not in ("cpdag", "pag", "pdag")
            or any(not isinstance(n, str) or not n for n in nodes)
            or len(set(nodes)) != len(nodes)
        ):
            raise ValueError("Invalid partial graph class or variable names")
        edges = tuple(sorted(self.edges))
        if any(e.left not in nodes or e.right not in nodes for e in edges) or len(
            {(e.left, e.right) for e in edges}
        ) != len(edges):
            raise ValueError("Duplicate edges or unknown endpoints")
        if self.kind == "cpdag" and any(
            "circle" in (e.left_mark, e.right_mark)
            or (e.left_mark, e.right_mark) == ("arrow", "arrow")
            for e in edges
        ):
            raise ValueError("CPDAGs support directed and undirected edges only")
        object.__setattr__(self, "nodes", tuple(sorted(nodes)))
        object.__setattr__(self, "edges", edges)

    @property
    def directed(self):
        """Return tail-to-arrow edges; PAG arrows encode ancestry rather than necessarily direct causes."""
        return tuple(
            (e.left, e.right) if e.left_mark == "tail" else (e.right, e.left)
            for e in self.edges
            if {e.left_mark, e.right_mark} == {"tail", "arrow"}
        )

    @property
    def unresolved(self):
        """Return edges containing circles or two tails."""
        return tuple(
            e
            for e in self.edges
            if "circle" in (e.left_mark, e.right_mark)
            or (e.left_mark, e.right_mark) == ("tail", "tail")
        )

    def to_dag(self):
        """Convert only a fully directed CPDAG; never invent orientations or convert a PAG."""
        if self.kind != "cpdag" or len(self.directed) != len(self.edges):
            raise ValueError(
                "Only a fully directed CPDAG specifies a unique DAG under the discovery assumptions"
            )
        return DAG(self.nodes, self.directed)

    def to_dict(self):
        """Return the graph class and endpoint-preserving edge records."""
        return {
            "kind": self.kind,
            "nodes": list(self.nodes),
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_grapl(self, path=None, *, title="CGraph", node_names=None):
        """Export a fully directed CPDAG; native GRAPL cannot encode CPDAG/PAG uncertainty."""
        if self.kind != "cpdag" or len(self.directed) != len(self.edges):
            raise ValueError(
                "Native GRAPL cannot represent unresolved CPDAG/PAG graphs; only a fully directed CPDAG can be exported without changing its meaning"
            )
        return self.to_dag().to_grapl(path, title=title, node_names=node_names)

    def to_dot(self):
        """Export Graphviz notation without requiring Graphviz or executing a renderer."""
        marks = {"tail": "none", "arrow": "normal", "circle": "odot"}
        lines = ["digraph CGraph {"]
        lines.extend(f"  {json.dumps(n)};" for n in self.nodes)
        for e in self.edges:
            lines.append(
                f"  {json.dumps(e.left)} -> {json.dumps(e.right)} [dir=both, arrowtail={marks[e.left_mark]}, arrowhead={marks[e.right_mark]}];"
            )
        return "\n".join([*lines, "}"]) + "\n"


@dataclass(frozen=True)
class DiscoveryResult:
    """An estimated graph with explicit assumptions, CI evidence, and completion status."""

    status: str
    graph: PartialGraph | None
    method: str
    assumptions: tuple[str, ...]
    diagnostics: Mapping = field(default_factory=dict)
    tests: tuple[Mapping, ...] = ()

    def __post_init__(self):
        if self.status not in (
            "completed",
            "inconsistent",
            "resource_limited",
            "insufficient_data",
            "error",
        ):
            raise ValueError("Unknown discovery status")
        if self.method not in ("pc", "fci") or not isinstance(self.diagnostics, Mapping):
            raise ValueError("Invalid discovery method or diagnostic record")
        if any(not isinstance(a, str) or not a for a in self.assumptions):
            raise ValueError("Assumptions must be nonempty descriptions")
        if self.graph is not None:
            if (
                not isinstance(self.graph, PartialGraph)
                or (self.method == "pc" and self.graph.kind == "pag")
                or (self.method == "fci" and self.graph.kind != "pag")
            ):
                raise ValueError("Graph class does not match the discovery method")
            if self.status == "completed" and self.graph.kind == "pdag":
                raise ValueError("An inconsistent PDAG is not a completed equivalence graph")
        if self.status == "completed" and self.graph is None:
            raise ValueError("Completed discovery requires a graph")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))
        object.__setattr__(self, "tests", tuple(freeze(t) for t in self.tests))

    def to_dict(self):
        """Return a detached, finite JSON record without raw observations."""
        return {
            "schema_version": "1",
            "status": self.status,
            "graph": self.graph.to_dict() if self.graph else None,
            "method": self.method,
            "assumptions": list(self.assumptions),
            "diagnostics": thaw(self.diagnostics),
            "tests": thaw(self.tests),
        }

    def to_json(self, path=None):
        """Serialize discovery evidence, optionally writing it atomically."""
        content = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        if path is not None:
            atomic_write(path, content)
        return content

    def to_grapl(self, path=None, *, title="CGraph", node_names=None):
        """Write a completed discovery graph when native GRAPL can preserve its semantics."""
        if self.status != "completed" or self.graph is None:
            raise ValueError("GRAPL export requires a completed discovery result")
        return self.graph.to_grapl(path, title=title, node_names=node_names)

    @classmethod
    def from_json(cls, content):
        """Load the strict discovery record schema without executing model code."""

        def reject(value):
            raise ValueError(f"Invalid JSON constant: {value}")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result

        value = json.loads(content, parse_constant=reject, object_pairs_hook=unique)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != "1"
            or set(value) != {"status", "graph", "method", "assumptions", "diagnostics", "tests"}
        ):
            raise ValueError("Invalid discovery schema")
        try:
            graph = value["graph"]
            if graph is not None:
                if not isinstance(graph, dict) or set(graph) != {"nodes", "edges", "kind"}:
                    raise ValueError("Invalid partial graph schema")
                value["graph"] = PartialGraph(
                    tuple(graph["nodes"]),
                    tuple(PartialEdge(**e) for e in graph["edges"]),
                    graph["kind"],
                )
            return cls(**value)
        except (TypeError, KeyError) as exc:
            raise ValueError("Invalid discovery record") from exc


def _prepare(data, names, max_variables):
    import numpy as np

    if hasattr(data, "columns") and names is None:
        names = tuple(data.columns)
    if hasattr(data, "to_numpy"):
        data = data.to_numpy()
    if np.iscomplexobj(data):
        raise ValueError("Complex-valued observations are unsupported")
    try:
        values = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Supply complete numeric continuous observations") from exc
    if (
        values.ndim != 2
        or values.shape[1] < 2
        or values.shape[0] < 8
        or values.shape[1] > max_variables
    ):
        raise ValueError("Require at least 8 rows and 2 variables within max_variables")
    if values.nbytes > 64 * 1024 * 1024 or not np.isfinite(values).all():
        raise ValueError("Data must be finite and at most 64 MiB")
    names = tuple(f"X{i + 1}" for i in range(values.shape[1])) if names is None else tuple(names)
    if (
        len(names) != values.shape[1]
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) or not n for n in names)
    ):
        raise ValueError("Supply one unique nonempty name per column")
    if any(len(np.unique(values[:, i])) < 3 for i in range(values.shape[1])):
        raise ValueError(
            "Fisher-z discovery requires continuous variables; binary/constant columns are unsupported"
        )
    order = sorted(range(len(names)), key=lambda i: names[i])
    values = values[:, order].copy()
    scale = values.std(axis=0)
    if not np.isfinite(scale).all() or (scale == 0).any():
        raise ValueError("Degenerate or numerically unrepresentable variation")
    values = (values - values.mean(axis=0)) / scale
    if not np.isfinite(values).all():
        raise ValueError("Numerical overflow during standardisation")
    return values, tuple(names[i] for i in order)


@lru_cache(maxsize=1)
def _plot_cache():
    """Keep backend font metadata in one private temporary directory per parent process."""
    return tempfile.TemporaryDirectory(prefix="cgraph-plot-cache-")


def discover(
    data,
    *,
    names=None,
    method="pc",
    alpha=0.01,
    max_tests=10000,
    timeout=30.0,
    max_conditioning=None,
    max_variables=100,
):
    """Estimate a CPDAG with PC-stable or a PAG with FCI from IID continuous observations.

    Fisher-z tests assume a jointly Gaussian distribution. PC additionally assumes
    causal sufficiency; FCI allows latent common causes. Both require acyclicity,
    Markov and faithfulness assumptions. Finite samples can lead to incorrect or
    inconsistent orientations. The output is an estimated equivalence class,
    never a certificate of causal truth or a posterior over graphs.

    Discovery runs in an isolated subprocess with a hard elapsed-time limit.
    CI-test and conditioning budgets abort without returning a purported complete
    graph. Existing factorisation and identification functions accept a DAG/ADMG,
    not an unresolved discovery output.
    """
    if method not in ("pc", "fci") or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("Use pc/fci and alpha in (0,1)")
    if (
        type(max_tests) is not int
        or max_tests < 1
        or type(max_variables) is not int
        or max_variables < 2
    ):
        raise ValueError("Test and variable budgets must be positive integers")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    if max_conditioning is not None and (type(max_conditioning) is not int or max_conditioning < 0):
        raise ValueError("max_conditioning must be None or a nonnegative integer")
    values, names = _prepare(data, names, max_variables)
    assumptions = (
        "IID observational units",
        "Jointly Gaussian variables and valid Fisher-z CI tests",
        "Acyclic causal structure, Markov condition and faithfulness",
        "No latent common causes or selection bias"
        if method == "pc"
        else "Latent common causes allowed; output marks describe ancestral equivalence",
    )
    request = {
        "data": values.tolist(),
        "names": names,
        "method": method,
        "alpha": alpha,
        "max_tests": max_tests,
        "max_conditioning": max_conditioning,
    }
    environment = os.environ.copy()
    environment.update(
        {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    )
    environment.update({"MPLBACKEND": "Agg", "MPLCONFIGDIR": _plot_cache().name})
    diagnostics = {
        "n_samples": len(values),
        "n_variables": len(names),
        "alpha": alpha,
        "test": "fisherz",
        "max_tests": max_tests,
        "max_conditioning": max_conditioning,
        "timeout": timeout,
        "alpha_scope": "Individual adaptive CI tests; no graph-level error probability",
    }
    try:
        process = subprocess.run(
            [sys.executable, "-m", "cgraph._discovery_worker"],
            input=json.dumps(request, allow_nan=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return DiscoveryResult(
            "resource_limited",
            None,
            method,
            assumptions,
            diagnostics | {"reason": "Hard subprocess timeout; partial evidence unavailable"},
        )
    if process.returncode != 0:
        return DiscoveryResult(
            "error",
            None,
            method,
            assumptions,
            diagnostics
            | {
                "reason": "Discovery worker exited unexpectedly",
                "worker_stderr": process.stderr[-2000:],
            },
        )
    try:
        result = json.loads(process.stdout)
        graph = result.get("graph")
        partial = (
            PartialGraph(
                tuple(graph["nodes"]),
                tuple(PartialEdge(**e) for e in graph["edges"]),
                graph["kind"],
            )
            if graph
            else None
        )
        return DiscoveryResult(
            result["status"],
            partial,
            method,
            assumptions,
            diagnostics | result["diagnostics"],
            tuple(result["tests"]),
        )
    except (ValueError, KeyError, TypeError) as exc:
        return DiscoveryResult(
            "error",
            None,
            method,
            assumptions,
            diagnostics | {"reason": f"Invalid discovery worker output: {exc}"},
        )
