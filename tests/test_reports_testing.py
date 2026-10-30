"""Evidence serialization, escaping, multiplicity, and statistical applicability."""

import numpy as np
import pandas as pd
import pytest

from cgraph import DAG
from cgraph.reports import AuditReport, CheckResult, Policy
from cgraph.testing import GaussianPartialCorrelation, audit_implications, holm


def test_report_immutability_roundtrip_and_policy(tmp_path):
    report = AuditReport(
        DAG(("x", "y")).to_dict(),
        {"outcome": "y"},
        {"status": "identified"},
        (
            CheckResult(
                "<script>",
                "completed",
                "statistical_test",
                "<script>alert(1)</script>",
                "inconclusive",
            ),
        ),
    )
    assert "<script>" not in report.to_html(tmp_path / "audit.html")
    assert "&lt;script&gt;" in report.to_html()
    assert AuditReport.from_json(report.to_json()).to_dict() == report.to_dict()
    with pytest.raises(TypeError):
        report.graph["nodes"] = []
    assert report.evaluate().decision == "block"
    assert report.evaluate(Policy(require_estimate=False)).decision == "review_required"
    with pytest.raises(ValueError):
        AuditReport.from_json(
            report.to_json().replace('"schema_version": "1"', '"schema_version": "99"')
        )
    with pytest.raises(ValueError):
        CheckResult("x", "unsupported", "statistical_test", "unsupported", "no_violation_detected")


def test_holm_and_gaussian_applicability():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.01], family_size=3) == pytest.approx([0.03])
    rng = np.random.default_rng(42)
    z = rng.normal(size=3000)
    data = pd.DataFrame({"x": z + rng.normal(size=3000), "y": z + rng.normal(size=3000), "z": z})
    test = GaussianPartialCorrelation()
    schema = dict.fromkeys(data.columns, "continuous_gaussian")
    assert test.test(data, "x", "y", (), schema)["p_value"] < 1e-10
    assert test.test(data, "x", "y", ("z",), schema)["p_value"] > 0.05
    assert test.test(data, "x", "y", (), {})["status"] == "unsupported"
    graph = DAG(tuple(data.columns), (("z", "x"), ("z", "y")))
    checks = audit_implications(graph, data, schema)
    assert checks[-1].metrics["completed"] == 1
    saturated = DAG(("x", "y"), (("x", "y"),))
    assert audit_implications(saturated, data, schema)[-1].finding == "inconclusive"
