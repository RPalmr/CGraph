# CGraph

CGraph learns causal graph structures from observational data. It uses conditional-independence tests to estimate connections and edge directions, preserving relationships the observations cannot resolve.

## From data to graphs

Discovery tests whether variables remain dependent after conditioning on other variables:

$$
X \perp Y \mid S.
$$

For jointly Gaussian observations, this corresponds to zero partial correlation, $\rho_{XY\cdot S}=0$. The tests guide adjacency removal and orientation rules.

| Method | Assumption about hidden causes | Estimated graph |
| --- | --- | --- |
| PC-stable | No unmeasured common causes | CPDAG: a class of observationally equivalent DAGs |
| FCI | Unmeasured common causes allowed | PAG: partial ancestral relationships across compatible models |

Both require independent observations, acyclicity, the causal Markov condition and faithfulness. Finite samples can produce incorrect graphs. Results are estimates under these assumptions, not proofs of causal truth.

## Example

```python
import numpy as np
from cgraph import discover

rng = np.random.default_rng(42)
a, b = rng.normal(size=(2, 5000))
c = a + b + rng.normal(size=5000)
data = np.column_stack((a, b, c))

result = discover(data, names=("A", "B", "C"), method="pc")
print(result.status)
for edge in result.graph.edges:
    print(edge)
```

This example recovers `A --> C <-- B`. The graph is learned from observations; it is not supplied as input. Use `method="fci"` to allow hidden common causes.

Endpoint marks preserve the remaining ambiguity:

- `A --> B`: directed edge in a CPDAG; in a PAG, a tail-to-arrow edge asserts ancestry rather than necessarily a direct cause.
- `A --- B`: unresolved direction in a CPDAG.
- `A o-> B` or `A o-o B`: unresolved ancestral endpoint marks in a PAG.
- `A <-> B`: neither endpoint is an ancestor of the other under the PAG assumptions; compatible with latent confounding. This is not two directed arrows.

A two-variable association generally cannot distinguish `A -> B` from `A <- B`. CGraph retains that uncertainty.

## Evidence and limits

`result` contains the estimated graph, assumptions, CI-test records, backend version and completion status. `alpha` is the threshold for individual adaptive tests, not a graph-level confidence probability. Variable names are sorted before discovery so input column order does not choose directions.

```python
result.to_json("artifacts/discovery.json")
dot = result.graph.to_dot()
```

`timeout`, `max_tests` and `max_conditioning` bound execution. A timeout or exhausted budget returns `resource_limited` without claiming a complete graph. Singular required tests return `insufficient_data`. PC orientations that fail CPDAG consistency checks return `inconsistent` with a general PDAG.

The current numerical path supports complete, finite, continuous numeric data and Fisher-z tests under joint Gaussianity. Binary, constant and missing inputs are rejected; mixed, categorical, nonlinear and time-series discovery need different tests or assumptions and are outside this release. Numeric categorical codes with more than two levels are not automatically recognised as categorical; do not use them with Fisher-z.

Only a fully directed CPDAG can be converted with `result.graph.to_dag()`. An unresolved CPDAG or a PAG is never silently converted into a causal DAG. `factorize`, `intervene` and `identify` remain available for reasoning under an explicit DAG/ADMG; they do not identify effects uniformly over an unresolved discovery result.

## Install and run

Python 3.11+. Discovery uses pinned [causal-learn](https://github.com/py-why/causal-learn) implementations of [PC-stable](https://jmlr.org/papers/volume15/colombo14a/colombo14a.pdf) and [FCI](https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html).

```bash
pip install 'cgraph-audit @ git+https://github.com/RPalmr/CGraph.git'
```

From a repository checkout:

```bash
uv sync --dev --locked
uv run python examples/discover_graphs.py
uv run cgraph discover --data artifacts/discovery/collider.csv --method pc --out artifacts/discovery/result.json --dot artifacts/discovery/result.dot
uv run pytest -q
```

The example demonstrates a collider, an unresolved direction and a latent common cause. Discovery runs locally in an isolated subprocess. Graph reports omit raw observations; the example explicitly writes its synthetic CSV. Tests cover known causal structures, endpoint conversion, column permutations, singular inputs and resource limits. Private backend font caches are temporary and reused within the calling process.

MIT licensed; see [LICENSE](LICENSE).
