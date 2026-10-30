"""End-to-end auditing and prevention of invalid causal estimation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from cgraph import ADMG, DAG, AuditConfig, CausalAudit, CausalQuery
from cgraph.estimation import AIPW


def data(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=500)
    t = rng.binomial(1, 0.5, size=500)
    return pd.DataFrame({"z": z, "t": t}), pd.Series(1.5 * t + z + rng.normal(size=500))


def test_end_to_end_and_frozen_holdout():
    X, y = data(1)
    graph = DAG(("z", "t", "y"), (("z", "t"), ("z", "y"), ("t", "y")))
    audit = CausalAudit(
        graph,
        CausalQuery.ate(treatment="t", outcome="y"),
        config=AuditConfig(placebo_repetitions=2, subset_repetitions=2),
    ).fit(X, y, assumptions=("IID units", "Graph and consistency assumed"))
    report = audit.audit_report()
    assert report.estimate is not None
    assert abs(report.estimate["value"] - 1.5) < 0.3
    assert len([c for c in report.checks if c.evidence_type == "stress_scenario"]) == 2
    new = audit.audit(*data(2))
    assert new.provenance["phase"] == "heldout_frozen_nuisance"
    assert audit.audit_report() is report
    assert clone(audit).graph == graph
    with pytest.raises(ValueError, match="training dataset"):
        audit.audit(X, y)


def test_nonidentification_and_wrong_adjustment_block_estimation():
    X, y = data(3)
    graph = ADMG(("z", "t", "y"), (("z", "t"), ("z", "y"), ("t", "y")), (("t", "y"),))
    audit = CausalAudit(graph, CausalQuery.ate(treatment="t", outcome="y")).fit(X, y)
    assert audit.estimator_ is None
    assert audit.audit_report().evaluate().decision == "block"
    dag = DAG(graph.nodes, graph.directed)
    wrong = CausalAudit(dag, audit.query, estimator=AIPW(adjustment=())).fit(X, y)
    assert wrong.estimator_ is None
    assert wrong.audit_report().identification["status"] == "identified"
    with pytest.raises(ValueError, match="independent"):
        CausalAudit(dag, audit.query, study_design="clustered").fit(X, y)
