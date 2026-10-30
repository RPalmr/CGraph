# CGraph

CGraph uses causal graphs to describe how variables influence one another and what happens when you intervene. It determines which intervention effects can be identified from observations under the graph's assumptions.

Three operations: **factorisation, intervention and identification**. Python 3.11+, with no runtime dependencies.

## Factorisation

For a fully observed causal DAG $G$, assume the Markov factorisation:

$$
P(\mathbf{v}) = \prod_i P(v_i \mid \mathrm{pa}_i^G).
$$

`factorize(graph)` constructs this expression from the parents of each variable.

## Intervention

Setting $T=t$ removes its incoming edges and replaces its mechanism with a fixed value. Truncated factorisation gives the distribution over the remaining variables:

$$
P(\mathbf{v}_{-T} \mid do(T=t))
= \left.\prod_{V_i \ne T} P(v_i \mid \mathrm{pa}_i^G)\right|_{T=t}.
$$

`intervene(graph, {"T": t})` constructs this distribution. It requires a fully observed causal DAG; conditioning on $T=t$ is generally different from intervening.

## Identification

For an observed adjustment set $Z$ containing no descendants of $T$ and blocking every back-door path:

$$
P(y \mid do(t)) = \sum_z P(y \mid t,z)P(z).
$$

`identify(graph, treatment="T", outcome="Y")` searches for such a set using graph separation. If adjustment does not succeed, it applies the [Shpitser–Pearl ID algorithm](https://escholarship.org/uc/item/9598x714). ID also supports ADMGs, whose bidirected edges represent hidden common causes.

The result contains an identifying expression, a verified non-identification witness, or a resource-limit status. Failure to find an adjustment set alone does not establish non-identifiability.

## Example

```python
from cgraph import DAG, factorize, intervene, identify

graph = DAG(
    nodes=("Z", "T", "Y"),
    directed=(("Z", "T"), ("Z", "Y"), ("T", "Y")),
)

joint = factorize(graph)
post = intervene(graph, {"T": 1}).marginalize("Y")
result = identify(graph, treatment="T", outcome="Y")

print(result.adjustment)
print(result.expression)
```

The engine selects $Z$ and returns $\sum_z P(Y\mid T,z)P(z)$. For continuous variables, marginalisation denotes integration. Numerical evaluation supports finite probability tables through `DiscreteDistribution`; continuous expressions remain symbolic.

All results depend on the supplied causal graph and relevant support conditions. CGraph constructs mathematical expressions; it does not verify causal assumptions from data.

## Install and run

```bash
pip install 'cgraph-audit @ git+https://github.com/RPalmr/CGraph.git'
python examples/three_functions.py
```

Run the example from a repository checkout. For development:

```bash
uv sync --dev --locked
uv run pytest -q
```

MIT licensed; see [LICENSE](LICENSE).
