"""Reproducible local performance measurements; results are hardware-specific."""

import json
import platform
import statistics
import time

from cgraph import DAG, CausalQuery, identify
from cgraph.reports import atomic_write


def measure(fn, repetitions=20):
    """Measure repeated completed operations without hiding exceptions."""
    elapsed = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        elapsed.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(elapsed),
        "maximum_seconds": max(elapsed),
        "repetitions": repetitions,
    }


def main():
    """Benchmark graph validation, repeated separation, and representative identification."""
    nodes = tuple(f"v{i:04}" for i in range(1000))
    edges = tuple(
        (nodes[i], nodes[i + step]) for i in range(1000) for step in range(1, 6) if i + step < 1000
    )
    graph = DAG(nodes, edges)
    confounded = DAG(("z", "t", "y"), (("z", "t"), ("z", "y"), ("t", "y")))
    query = CausalQuery.interventional(treatment="t", outcome="y")
    results = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nodes": len(nodes),
        "edges": len(edges),
        "validation": measure(lambda: DAG(nodes, edges)),
        "separation": measure(lambda: graph.separated((nodes[0],), (nodes[-1],), nodes[450:455])),
        "identification_small": measure(lambda: identify(confounded, query)),
    }
    content = json.dumps(results, indent=2) + "\n"
    atomic_write("artifacts/benchmarks/local.json", content)
    print(content)


if __name__ == "__main__":
    main()
