# CGraph

CGraph implements three mathematical operations on causal graphs:

1. `factorize`: construct a DAG's Markov factorization.
2. `intervene`: construct the truncated distribution after fixing intervention values.
3. `identify`: derive an interventional distribution from observational probabilities using back-door adjustment or the general ID algorithm.

There are no third-party runtime dependencies. Python 3.11–3.14 is tested on Linux and macOS.

## Install

```bash
pip install 'cgraph-audit @ git+https://github.com/RPalmr/CGraph.git'
```

The distribution name remains `cgraph-audit`; the Python import is `cgraph`. Version 0.2 removes the former auditing/estimation API. For reproducible deployment, pin a reviewed commit or install a built wheel.

## 1. Markov factorization

For a supplied DAG G, assume its joint observational law obeys:

$$P(\mathbf{v})=\prod_i P(v_i\mid\mathrm{pa}_i^G).$$

```python
from cgraph import DAG, factorize

graph = DAG(
    nodes=("Z", "T", "Y"),
    directed=(("Z", "T"), ("Z", "Y"), ("T", "Y")),
)
formula = factorize(graph)
print(formula)
```

```text
(P(Z)) * (P(T | Z)) * (P(Y | T, Z))
```

This constructs the factorization implied by the supplied graph. It assumes the Markov condition; it does not test it against data or establish that the graph is correct. Factorization into observed-parent mechanisms requires a fully observed causal DAG. Applying it directly to an observed-variable ADMG with latent confounding is rejected.

## 2. Intervention and truncated factorization

For a fully observed causal DAG with invariant mechanisms, intervention replaces the treatment mechanism with a fixed value:

$$P(\mathbf{v}_{-T}\mid do(T=t))=
\left.\prod_{V_i\ne T}P(v_i\mid\mathrm{pa}_i^G)\right|_{T=t}.$$

```python
from cgraph import intervene

post = intervene(graph, {"T": 1})
print(post.expression)
print(post.graph.directed)
outcome_distribution = post.marginalize("Y")
```

The expression retains `P(Z)` and `P(Y | T, Z)` and drops `P(T | Z)`. The returned graph removes incoming edges to T. The fixed value is stored separately in immutable `post.values` and applied during evaluation. Multiple simultaneous interventions are supported.

The formula is a distribution on the **nonintervened variables**. Its extension to the full joint distribution includes a point mass at the specified treatment value; incompatible fixed assignments evaluate to zero. Observational conditioning `P(Y | T=t)` generally differs from intervention `P(Y | do(T=t))`.

`intervene` rejects ADMG input: simply deleting observed treatment factors is not valid under general hidden confounding. Use identification for that setting.

## 3. Back-door adjustment and general identification

```python
from cgraph import identify

result = identify(graph, treatment="T", outcome="Y")
print(result.status)
print(result.method)
print(result.adjustment)
print(result.expression)
```

```text
identified
backdoor
('Z',)
marginalize[Z]((P(Y | T, Z)) * (P(Z)))
```

For one treatment, the automatic path searches for a minimum-cardinality observed set Z containing no treatment descendants and blocking all back-door paths. Its formula is:

$$P(y\mid do(t))=\sum_z P(y\mid t,z)P(z).$$

The search uses separation in the graph with outgoing treatment arrows removed. An empty valid set is returned as `()`; absence of a set is distinct from that result. Graph traversal uses cached immutable adjacency and an active-trail search.

Back-door adjustment is sufficient, not complete. If no back-door set is found, the engine runs the [Shpitser–Pearl ID algorithm](https://escholarship.org/uc/item/9598x714), following the Tian–Pearl identification line of work. ID supports joint treatment/outcome sets and observed-variable ADMGs with bidirected edges representing hidden common causes.

```python
from cgraph import ADMG, identify

frontdoor = ADMG(
    nodes=("T", "M", "Y"),
    directed=(("T", "M"), ("M", "Y")),
    bidirected=(("T", "Y"),),
)
result = identify(frontdoor, treatment="T", outcome="Y")
assert result.status == "identified"
assert result.method == "id"
```

This frontdoor model is identifiable despite lacking an observed back-door set. Conversely, a two-node model with both `T -> Y` and `T <-> Y` is not identifiable nonparametrically from its observational joint law alone.

Results distinguish:

- `identified`: an observational functional is returned.
- `not_identified`: ID returned a verified hedge for the failing recursive subproblem.
- `resource_limited`: a time, recursion, or computation budget prevented completion.

Invalid graphs and queries raise `ValueError`. Supply `method="id"` to bypass adjustment search. `max_sets`, `max_calls`, and `timeout` bound computation. If the adjustment search budget is exhausted before the time limit, automatic mode can still attempt ID. Time limits are cooperative, rather than hard process-isolation guarantees.

A fully observed causal DAG permits identification through the g-formula, subject to the relevant support conditions. Failure of a sufficient adjustment search alone is never reported as general nonidentification. Identification remains conditional on the supplied causal model and assumptions.

## Exact finite-distribution evaluation

No estimator fitting or statistical hypothesis testing is performed. Numerical evaluation accepts an explicitly supplied finite observational joint probability table:

```python
from cgraph import DiscreteDistribution

observed = DiscreteDistribution(
    nodes=("T", "Y"),
    masses={(0, 0): 0.4, (0, 1): 0.1, (1, 0): 0.15, (1, 1): 0.35},
)
simple = DAG(("T", "Y"), (("T", "Y"),))
result = identify(simple, treatment="T", outcome="Y")
assert abs(result.expression.evaluate(observed, {"T": 1, "Y": 1}) - 0.7) < 1e-12
```

Tables must be normalized and finite. Evaluating conditional factors outside observational support raises an error. Nonparametric identification is distinct from numerical support in a particular probability table. The caller is responsible for the table satisfying the supplied causal model's observable constraints. Continuous formulas remain symbolic; `marginalize` denotes sums or integrals as appropriate, but a numerical continuous integrator is not included.

Expressions have text, LaTeX, and JSON representations. ID traces and nonidentification witnesses are immutable and serializable. Recursive kernels preserve their fixed-variable dependencies rather than being relabeled as ordinary observed conditionals. Expressions are not guaranteed to be algebraically minimal.

## Run the complete example

```bash
uv sync --dev --locked
uv run python examples/three_functions.py
```

The example constructs a known binary causal model and obtains `P(Y=1 | do(T=1)) = 0.650` through both truncated factorization and back-door identification. Tests also check that the observational conditional in that model is `0.740`.

## CLI

Given a graph JSON with `kind`, `nodes`, `directed`, and `bidirected`:

```bash
cgraph factorize --graph graph.json
cgraph intervene --graph graph.json --values '{"T":1}'
cgraph identify --graph graph.json --treatment T --outcome Y
```

Each command supports `--out formula.json`. JSON parsing rejects duplicate keys and nonfinite constants. No arbitrary configuration code is executed and no network access occurs. Exit codes are 0 for a completed computation (including mathematical nonidentification), 2 for invalid input, 4 for resource exhaustion, and 5 for unexpected execution failure.

## Development and verification

```bash
uv sync --dev --locked
uv run ruff check src tests examples benchmarks scripts
uv run ruff format --check src tests examples benchmarks scripts
uv run pytest -q
uv build
uv run python scripts/verify_release.py
```

Tests compare separation with independent graph oracles, check factorization and intervention normalization against exact probability tables, and compare ID expressions with exact interventions in generated latent-variable causal models. Frontdoor, confounded bows, joint interventions, support errors, and resource limits have explicit tests. CI builds and installs the wheel in a clean environment.

Python source uses docstrings without inline comment tokens. MIT licensed; see [LICENSE](LICENSE).
