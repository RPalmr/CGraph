# CGraph

CGraph audits causal claims under explicit assumptions. It combines deterministic graph identification with statistical diagnostics, causal estimation, and sensitivity analysis. It runs locally and makes no network requests during an audit.

**An identified effect is conditional on the supplied causal model. Passing statistical checks does not establish that the graph is true or that hidden confounding is absent.** Deployment policy decisions refer to the configured checks, not to a probability of causal validity.

## Installation

Requires Python 3.11 or later. Install from the repository; a PyPI release is not assumed:

```bash
pip install 'cgraph-audit[stats] @ git+https://github.com/RPalmr/CGraph.git'
```

For reproducible deployment, replace the repository URL's default revision with a reviewed commit SHA, or install a built wheel. The graph, identification, expression, and report layers have no third-party runtime dependencies. The `stats` extra installs NumPy, SciPy, pandas, and scikit-learn.

```bash
git clone https://github.com/RPalmr/CGraph.git
cd CGraph
uv sync --dev --locked
uv run pytest -q
uv build
```

## Symbolic identification

```python
from cgraph import ADMG, CausalQuery, identify

graph = ADMG(
    nodes=("treatment", "mediator", "outcome"),
    directed=(("treatment", "mediator"), ("mediator", "outcome")),
    bidirected=(("treatment", "outcome"),),
)
query = CausalQuery.interventional(
    treatment="treatment",
    outcome="outcome",
)
result = identify(graph, query)
print(result.status)
print(result.expression)
```

This frontdoor model is identifiable despite latent treatment–outcome confounding. Identification supports joint treatment/outcome sets and emits a scoped symbolic expression and derivation trace. A directed/bidirected bow is allowed in an ADMG; a directed cycle is rejected.

`identify` implements the [Shpitser–Pearl ID algorithm](https://escholarship.org/uc/item/9598x714). A nonidentification result includes a validated hedge for the failing recursive subproblem. Computational exhaustion returns `resource_limited`, never `not_identified`. Expressions may retain redundant conditioning terms and are not guaranteed to be algebraically minimal.

```python
from cgraph import identify_conditional

conditional = identify_conditional(graph, query, given=("mediator",))
```

Conditional intervention distributions use [IDC](https://ftp.cs.ucla.edu/pub/stat_ser/r329-uai.pdf), including observation-to-intervention exchanges before normalization. Expressions use a generic marginalization operator; finite discrete distributions can be evaluated through `cgraph.symbolic.DiscreteDistribution`. General numerical compilation of ID/IDC functionals is not provided.

## Statistical auditing

```python
from cgraph import DAG, AuditConfig, CausalAudit, CausalQuery
from cgraph.estimation import AIPW

graph = DAG(
    nodes=("experience", "training", "productivity"),
    directed=(
        ("experience", "training"),
        ("experience", "productivity"),
        ("training", "productivity"),
    ),
)
model = CausalAudit(
    graph=graph,
    query=CausalQuery.ate(
        treatment="training",
        outcome="productivity",
        control=0,
        treated=1,
        target_population="Eligible employees represented by the cohort",
    ),
    estimator=AIPW(adjustment=("experience",), cv=5),
    config=AuditConfig(placebo_repetitions=20, subset_repetitions=20),
    random_state=42,
)
model.fit(
    X_train,
    y_train,
    assumptions=(
        "The supplied graph and its omitted edges are assumed correct",
        "Independent units, consistent treatment, and no interference",
        "Overlap and nuisance-estimation regularity conditions",
    ),
    dataset_id="training-cohort-v1",
)
report = model.audit_report()
report.to_json("artifacts/audit.json")
report.to_html("artifacts/audit.html")
print(report.evaluate().to_dict())

holdout_report = model.audit(X_holdout, y_holdout, dataset_id="holdout-v1")
```

`X` must be a DataFrame containing treatment and observed covariates; `y` supplies the outcome separately. All supplied column names must appear in the graph. Indices must be unique and aligned. Estimation requires complete, finite numeric adjustment variables and a finite scalar outcome. Treatment levels are explicitly specified numeric values.

`fit` uses fold-local preprocessing and cross-fitted nuisance estimates, then fits final nuisance models for held-out scoring. `audit_report()` returns the immutable training snapshot. `audit()` never refits and returns a separate report. Holdout independence and population comparability remain caller responsibilities; exact reuse of the training frame is rejected. Changing parameters through `set_params` invalidates fitted evidence.

The built-in estimator supports binary-treatment **mean differences**, including risk differences for binary outcomes. It does not turn predictive coefficients or feature importances into causal effects. Custom scikit-learn nuisance learners can be supplied through `AIPW(propensity_model=..., outcome_model=...)`; their preprocessing must be part of the learner pipeline. Missingness, numeric overflow, and unsupported propensity scores are explicit errors. Propensities are never silently clipped and rows are never silently trimmed.

Run a complete synthetic example:

```bash
python examples/basic_audit.py
```

## Empirical checks and their interpretation

Graph-implied tests are generated from pairwise local Markov implications of a fixed DAG. The test family is prespecified from the graph and corrected with Holm's method. Saturated graphs may have no testable implications; those reports remain empirically inconclusive.

Gaussian tests require an explicit schema value `continuous_gaussian` for every variable in that particular test. This declares a jointly Gaussian subvector; it is not inferred from a numeric dtype. Binary or constant observed variables are not accepted by the Gaussian test. A conservative inferred schema leaves unsupported tests visible.

For discrete data:

```python
from cgraph import AuditConfig
from cgraph.testing import DiscreteConditionalIndependence

config = AuditConfig(ci_test=DiscreteConditionalIndependence(min_expected=5))
schema = {"treatment": "binary", "outcome": "binary", "group": "categorical"}
```

The discrete test uses stratified likelihood-ratio statistics and requires adequate expected cell counts. Sparse strata yield `insufficient_data`. Neither method is a universal conditional-independence test. Automatic empirical implication generation is currently DAG-only; ADMG separation queries remain available through `graph.separated(...)`.

Synthetic placebo tests replace treatment with independent pseudo-treatment and refit the full pipeline. They diagnose a synthetic task, not observational treatment randomization. Subsample checks describe stability across overlapping subsets; their quantiles are not confidence intervals. Neither check rules out hidden confounding.

## Sensitivity analysis

```python
from cgraph.sensitivity import RiskRatio, evalue, rosenbaum_pairs

result = evalue(
    RiskRatio(
        value=2.0,
        confidence_interval=(1.3, 3.1),
        adjustment=("age",),
        outcome_event="event=1",
        contrast="treated/control",
        scope="conditional",
    )
)

bounds = rosenbaum_pairs(
    outcome=[3.0, 1.0, 1.0, 2.0],
    treatment=[1, 0, 0, 1],
    pair_ids=["pair-a", "pair-a", "pair-b", "pair-b"],
    gammas=(1.0, 1.5, 2.0, 3.0),
    alternative="greater",
)
```

The E-value implementation accepts an explicitly described **conditional risk ratio**, following [VanderWeele and Ding](https://dash.harvard.edu/entities/publication/73120378-fbd5-6bd4-e053-0100007fdf3b). It rejects bare numbers, mean differences, and marginal-risk-ratio scope. It performs no odds-ratio or hazard-ratio conversions.

Rosenbaum bounds currently use an exact one-sided **paired sign test**, not a signed-rank test. Inputs require independent matched pairs with one treated and one control unit each. Zero differences are excluded and reported. Gamma constrains within-pair treatment-assignment odds; the output bounds a sharp-null p-value, not an ATE interval. These assumptions follow the [matched observational sensitivity framework](https://pmc.ncbi.nlm.nih.gov/articles/PMC3416023/).

Named graph scenarios expose structural sensitivity:

```python
from cgraph.scenarios import add_latent_confounding, audit_graph_scenarios

candidate = add_latent_confounding(graph, (("training", "productivity"),))
check = audit_graph_scenarios(
    graph,
    model.query,
    [("unmeasured_common_cause", candidate)],
)
```

`identify_family` audits an explicit finite graph family. It declares a common functional only when all enumerated expressions are structurally equal and enumeration is complete. Different expressions can still be algebraically equal; the library does not claim otherwise. Candidate fractions are not probabilities. `latent_projection` converts an explicit latent DAG into an observed-variable ADMG.

## Reports and deployment gates

Each check separates execution (`completed`, `unsupported`, `insufficient_data`, `resource_limited`, and others) from its scientific finding. Evidence is typed as a graph theorem, statistical test, estimator diagnostic, sensitivity analysis, stress scenario, or user assertion.

Reports preserve graph/query snapshots, assumptions, dependencies, seeds, test coverage, and fit configuration. They contain no raw rows by default. Local JSON/HTML writes use atomic replacement; HTML escapes graph labels and contains no scripts or remote assets. JSON is a data format, not a signed attestation or executable model format. Load reports only from trusted audit runs when making deployment decisions.

```python
from cgraph import Policy

policy = Policy(
    require_estimate=True,
    require_empirical=True,
    require_declared_assumptions=True,
)
decision = report.evaluate(policy)
```

The default policy blocks missing required causal estimates and nonidentified effects. Empirical contradictions, missing required evidence, and absent assumption declarations require review. `allow` means the configured policy requirements were satisfied; it does not mean causal truth was established. Policies for graph-only analysis can explicitly disable estimation and empirical requirements.

## Command line

```bash
cgraph identify --graph graph.json --query query.json --out identified.json
cgraph identify --graph graph.json --query query.json --given group --out conditional.json
cgraph audit --config audit-config.json --data cohort.csv --out artifacts/run
cgraph report artifacts/run/audit.json --html artifacts/run/audit.html
cgraph policy artifacts/run/audit.json --policy policy.json
```

`graph.json` uses `kind` (`dag` or `admg`), `nodes`, `directed`, and `bidirected`. Query JSON uses the `CausalQuery` field names. An audit configuration contains `graph`, `query`, optional `schema`, `assumptions`, `random_state`, `dataset_id`, `audit`, and `estimator`. `audit.ci_test` is `gaussian` or `discrete`; estimator settings support adjustment, folds, confidence level, and overlap threshold. Configuration never imports or evaluates arbitrary code.

Exit codes: `0` successful computation or allowed policy; `2` invalid input/dependency; `3` policy block; `4` review required or identification resource limit; `5` unexpected execution error. A successfully computed `not_identified` result exits `0` from `identify`. Enforcement is a separate `policy` command.

## Supported deployment scope

| Component | Supported | Outside current scope |
| --- | --- | --- |
| Graphs | DAGs, ADMGs, latent DAG projection, finite explicit graph families | PAG/CPDAG discovery, selection graphs, feedback cycles |
| Identification | Joint ID, conditional IDC, traces and hedge verification | Counterfactual/path-specific effects, transportability |
| Estimation | IID binary-treatment DAG-adjustment AIPW, numeric complete covariates | Longitudinal/interference/clustered inference, continuous treatment, arbitrary ID compilation |
| Tests | Declared Gaussian or sufficiently populated discrete DAG implications | Universal nonlinear/mixed-data CI testing, full latent-model constraints |
| Sensitivity | Conditional-RR E-values, paired sign-test Gamma bounds, graph scenarios | Generic confounding probabilities, unrestricted causal bounds |
| Integrations | Scikit-learn nuisance models, local Python/CLI, JSON/HTML | Automatic discovery and third-party causal-estimator adapters |

Statistical validity remains conditional on method-specific assumptions. Automated tests and simulation coverage are evidence about the implementation and simulated settings, not a substitute for reviewing a real study's design. Use graph-only workflows when the statistical scope does not match your dataset.

## Verification and development

```bash
uv run ruff check src tests examples benchmarks scripts
uv run ruff format --check src tests examples benchmarks scripts
uv run pytest -q
uv run python benchmarks/run.py
```

Tests compare separation with independent NetworkX oracles, evaluate identified expressions against exact latent-SCM interventions, verify hedge certificates, exercise immutable reports and CLI failures, and check statistical calibration with prespecified Monte Carlo tolerances. CI tests Python 3.11–3.14 on Linux and macOS, builds artifacts, and tests the dependency-free core separately.

Benchmarks write hardware-specific measurements to local `artifacts/`. Basic graph operations use cached immutable adjacency and active-trail traversal; symbolic recursion is memoized. Identification deadlines are cooperative checks at algorithm boundaries, not hard process-isolation guarantees. Use an external process/container resource limit when hard deadlines are required. Statistical test counts and resampling counts are bounded by configuration.

Python code uses docstrings without inline comment tokens. MIT licensed; see [LICENSE](LICENSE).
