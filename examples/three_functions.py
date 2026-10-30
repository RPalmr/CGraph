"""Demonstrate the three mathematical operations on a known finite causal model."""

import itertools

from cgraph import DAG, DiscreteDistribution, factorize, identify, intervene


def main():
    """Compare Markov factorization, truncated intervention, and back-door identification."""
    graph = DAG(("Z", "T", "Y"), (("Z", "T"), ("Z", "Y"), ("T", "Y")))
    masses = {}
    for z, t, y in itertools.product((0, 1), repeat=3):
        pt = 0.2 + 0.6 * z
        py = 0.1 + 0.3 * z + 0.4 * t
        masses[z, t, y] = 0.5 * (pt if t else 1 - pt) * (py if y else 1 - py)
    observed = DiscreteDistribution(("Z", "T", "Y"), masses)
    joint = factorize(graph)
    post = intervene(graph, {"T": 1}).marginalize(("Y",))
    result = identify(graph, treatment="T", outcome="Y")
    print("Factorization:", joint)
    print("Intervention:", post.to_dict()["formula"], "with T=1")
    print("Identification:", result.method, "adjustment:", result.adjustment)
    print("Identified formula:", result.expression)
    direct = post.evaluate(observed, {"Y": 1})
    identified = result.expression.evaluate(observed, {"T": 1, "Y": 1})
    assert abs(direct - 0.65) < 1e-12
    assert abs(direct - identified) < 1e-12
    print(f"P(Y=1 | do(T=1)) = {identified:.3f}")


if __name__ == "__main__":
    main()
