"""Learn causal graph classes from observations with known generating structures."""

from pathlib import Path

import numpy as np

from cgraph import discover
from cgraph._serialization import atomic_write


def save_result(label, data, names, method, output):
    """Run discovery, print endpoint marks, and write reproducible graph artifacts."""
    result = discover(data, names=names, method=method, timeout=60)
    result.to_json(output / f"{label}.json")
    if result.status != "completed":
        raise RuntimeError(result.to_json())
    atomic_write(output / f"{label}.dot", result.graph.to_dot())
    if result.graph.kind == "cpdag" and not result.graph.unresolved:
        result.to_grapl(output / f"{label}.grapl", title=label)
    print(f"{label}: {result.graph.kind}; {len(result.tests)} unique CI tests")
    for edge in result.graph.edges:
        print(f"  {edge}")
    return result


def main():
    """Demonstrate collider recovery, direction ambiguity, and hidden confounding."""
    output = Path("artifacts/discovery")
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(2, 5000))
    c = a + b + rng.normal(size=5000)
    data = np.column_stack((a, b, c))
    np.savetxt(output_csv(output), data, delimiter=",", header="A,B,C", comments="")
    pc = save_result("collider_pc", data, ("A", "B", "C"), "pc", output)
    assert set(pc.graph.directed) == {("A", "C"), ("B", "C")}
    save_result("collider_fci", data, ("A", "B", "C"), "fci", output)
    rng = np.random.default_rng(81)
    x = rng.normal(size=2000)
    y = x + rng.normal(size=2000)
    ambiguous = save_result(
        "unresolved_direction", np.column_stack((x, y)), ("X", "Y"), "pc", output
    )
    assert ambiguous.graph.unresolved
    rng = np.random.default_rng(117)
    a, b, latent = rng.normal(size=(3, 8000))
    x = a + latent + rng.normal(size=8000)
    y = b + latent + rng.normal(size=8000)
    hidden = save_result(
        "latent_confounding", np.column_stack((a, b, x, y)), ("A", "B", "X", "Y"), "fci", output
    )
    assert any((e.left_mark, e.right_mark) == ("arrow", "arrow") for e in hidden.graph.edges)
    print("Graph recovery in these examples does not certify recovery on arbitrary real data.")
    print(f"JSON, DOT, supported native GRAPL graphs and example CSV: {output}")


def output_csv(output):
    """Prepare an ignored local CSV path without storing observations in reports."""
    output.mkdir(parents=True, exist_ok=True)
    return output / "collider.csv"


if __name__ == "__main__":
    main()
