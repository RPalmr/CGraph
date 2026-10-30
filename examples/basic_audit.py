"""Run a reproducible causal audit on a synthetic cohort with known treatment effect."""

import numpy as np
import pandas as pd

from cgraph import DAG, AuditConfig, CausalAudit, CausalQuery
from cgraph.estimation import AIPW


def main():
    """Create local reports without transmitting data or loading external models."""
    rng = np.random.default_rng(42)
    experience = rng.normal(size=2000)
    treatment = rng.binomial(1, 1 / (1 + np.exp(-0.4 * experience)))
    outcome = 2 * treatment + experience + rng.normal(size=2000)
    X = pd.DataFrame({"experience": experience, "training": treatment})
    graph = DAG(
        ("experience", "training", "productivity"),
        (("experience", "training"), ("experience", "productivity"), ("training", "productivity")),
    )
    model = CausalAudit(
        graph,
        CausalQuery.ate(
            treatment="training", outcome="productivity", target_population="Synthetic IID cohort"
        ),
        estimator=AIPW(adjustment=("experience",)),
        config=AuditConfig(placebo_repetitions=3, subset_repetitions=3),
        random_state=42,
    )
    model.fit(
        X,
        pd.Series(outcome),
        assumptions=(
            "The supplied causal DAG is assumed correct",
            "Independent units, consistent treatment, no interference",
            "Overlap and AIPW nuisance regularity conditions",
        ),
        dataset_id="synthetic-v1",
    )
    report = model.audit_report()
    report.to_json("artifacts/example/audit.json")
    report.to_html("artifacts/example/audit.html")
    print(f"Estimated effect: {report.estimate['value']:.3f}; true effect: 2.000")
    print(
        f"Policy: {report.evaluate().decision}; no nontrivial CI implications in this saturated graph"
    )


if __name__ == "__main__":
    main()
