"""Isolated, bounded causal-learn execution with recorded conditional-independence tests."""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import itertools
import json
import sys
import warnings

import numpy as np
from causallearn.utils.cit import CIT_Base, FisherZ, register_ci_test


class _BudgetExceeded(Exception):
    """Discovery cannot complete within the requested CI budget."""


class _InsufficientData(Exception):
    """A required statistical test has no supported sampling regime."""


class _TrackedFisherZ(CIT_Base):
    """Cache and record valid Fisher-z tests while enforcing global execution budgets."""

    def __init__(self, data, tracker, **kwargs):
        super().__init__(data)
        self.method = "fisherz"
        self.test = FisherZ(data)
        self.tracker = tracker

    def __deepcopy__(self, memo):
        return self

    def __call__(self, x, y, condition_set=None):
        x, y = sorted((int(x), int(y)))
        z = tuple(sorted(int(i) for i in (() if condition_set is None else condition_set)))
        key = (x, y, z)
        tracker = self.tracker
        if key in tracker["cache"]:
            return tracker["cache"][key]
        if len(tracker["tests"]) >= tracker["max_tests"] or (
            tracker["max_conditioning"] is not None and len(z) > tracker["max_conditioning"]
        ):
            raise _BudgetExceeded("CI test count or conditioning budget exhausted")
        if len(self.data) <= len(z) + 3:
            raise _InsufficientData("Insufficient residual degrees of freedom for Fisher-z")
        correlation = self.test.correlation_matrix[np.ix_([x, y, *z], [x, y, *z])]
        if np.linalg.cond(correlation) > 1e12:
            raise _InsufficientData("Singular or ill-conditioned variables in a required CI test")
        p = float(self.test(x, y, z))
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise _InsufficientData("CI test produced an invalid p-value")
        tracker["cache"][key] = p
        names = tracker["names"]
        tracker["tests"].append(
            {
                "x": names[x],
                "y": names[y],
                "given": [names[i] for i in z],
                "p_value": p,
                "independence_not_rejected": p > tracker["alpha"],
            }
        )
        return p


def _cpdag_consistent(graph, names):
    """Verify a consistent extension and round-trip its completed equivalence graph."""
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    matrix = graph.graph
    if any(
        (matrix[i, j], matrix[j, i]) not in ((0, 0), (-1, -1), (-1, 1), (1, -1))
        for i, j in itertools.combinations(range(len(names)), 2)
    ):
        return False
    remaining = set(range(len(names)))
    directed = {
        (i, j) for i in remaining for j in remaining if matrix[i, j] == -1 and matrix[j, i] == 1
    }
    extension = set(directed)
    while remaining:
        chosen = None
        for i in sorted(remaining):
            if any(a == i and b in remaining for a, b in directed):
                continue
            neighbors = {j for j in remaining - {i} if matrix[i, j] == matrix[j, i] == -1}
            parents = {a for a, b in directed if b == i and a in remaining}
            if any(matrix[a, b] == 0 for a in neighbors for b in (neighbors | parents) - {a}):
                continue
            chosen = i
            extension.update((j, i) for j in neighbors)
            break
        if chosen is None:
            return False
        remaining.remove(chosen)
    nodes = [GraphNode(n) for n in names]
    dag = Dag(nodes)
    for i, j in sorted(extension):
        dag.add_directed_edge(nodes[i], nodes[j])
    return bool(np.array_equal(dag2cpdag(dag).graph, matrix))


def _ancestral_consistent(graph, names):
    """Reject compelled directed cycles and arrowheads that contradict compelled ancestry."""
    from .graph import DAG

    matrix = graph.graph
    directed = tuple(
        (names[i], names[j])
        for i in range(len(names))
        for j in range(len(names))
        if matrix[i, j] == -1 and matrix[j, i] == 1
    )
    try:
        dag = DAG(tuple(names), directed)
    except ValueError:
        return False
    for i, j in itertools.combinations(range(len(names)), 2):
        if matrix[i, j] == 1 and names[j] in dag.descendants((names[i],)):
            return False
        if matrix[j, i] == 1 and names[i] in dag.descendants((names[j],)):
            return False
        if matrix[i, j] == matrix[j, i] == -1 and (
            np.any(matrix[i, :] == 1) or np.any(matrix[j, :] == 1)
        ):
            return False
    return True


def run(request):
    """Run the backend and return partial marks without claiming a unique causal graph."""
    tracker = {
        "cache": {},
        "tests": [],
        **{k: request[k] for k in ("names", "max_tests", "max_conditioning", "alpha")},
    }
    diagnostics = {
        "backend": "causal-learn",
        "backend_version": importlib.metadata.version("causal-learn"),
        "warnings": [],
    }
    register_ci_test("cgraph_fisherz_v1", _TrackedFisherZ)
    try:
        data = np.asarray(request["data"], dtype=float)
        with (
            warnings.catch_warnings(record=True) as caught,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            warnings.simplefilter("always")
            if request["method"] == "pc":
                from causallearn.search.ConstraintBased.PC import pc

                graph = pc(
                    data,
                    alpha=request["alpha"],
                    indep_test="cgraph_fisherz_v1",
                    stable=True,
                    uc_rule=0,
                    uc_priority=2,
                    show_progress=False,
                    verbose=False,
                    node_names=request["names"],
                    tracker=tracker,
                ).G
                consistent = _cpdag_consistent(graph, request["names"])
                kind, status = ("cpdag", "completed") if consistent else ("pdag", "inconsistent")
                if not consistent:
                    diagnostics["warnings"].append(
                        "Finite-sample orientations do not form a valid CPDAG; no causal DAG may be inferred"
                    )
            else:
                from causallearn.search.ConstraintBased.FCI import fci

                graph, _ = fci(
                    data,
                    independence_test_method="cgraph_fisherz_v1",
                    alpha=request["alpha"],
                    show_progress=False,
                    verbose=False,
                    node_names=request["names"],
                    tracker=tracker,
                )
                kind, status = (
                    "pag",
                    "completed"
                    if _ancestral_consistent(graph, request["names"])
                    else "inconsistent",
                )
                if status == "inconsistent":
                    diagnostics["warnings"].append(
                        "Compelled endpoint marks violate ancestral constraints; interpret no causal conclusions"
                    )
            diagnostics["warnings"].extend(str(w.message) for w in caught)
        marks = {-1: "tail", 1: "arrow", 2: "circle"}
        edges = []
        names = request["names"]
        for i, j in itertools.combinations(range(len(names)), 2):
            a, b = int(graph.graph[i, j]), int(graph.graph[j, i])
            if a == b == 0:
                continue
            if a not in marks or b not in marks:
                return {
                    "status": "inconsistent",
                    "graph": None,
                    "diagnostics": diagnostics
                    | {"reason": "Unsupported conflicting endpoint marks"},
                    "tests": tracker["tests"],
                }
            edges.append(
                {"left": names[i], "right": names[j], "left_mark": marks[a], "right_mark": marks[b]}
            )
        result = {"status": status, "graph": {"nodes": names, "edges": edges, "kind": kind}}
    except _BudgetExceeded as exc:
        result = {"status": "resource_limited", "graph": None}
        diagnostics["reason"] = str(exc)
    except (_InsufficientData, np.linalg.LinAlgError) as exc:
        result = {"status": "insufficient_data", "graph": None}
        diagnostics["reason"] = str(exc)
    except Exception as exc:
        result = {"status": "error", "graph": None}
        diagnostics["reason"] = f"{type(exc).__name__}: {exc}"
    diagnostics["unique_ci_tests"] = len(tracker["tests"])
    return result | {"diagnostics": diagnostics, "tests": tracker["tests"]}


def main():
    """Read one numeric discovery request and emit only JSON on stdout."""
    request = json.load(sys.stdin)
    result = run(request)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
