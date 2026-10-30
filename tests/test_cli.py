"""CLI integration, documented exit codes, and strict configuration parsing."""

import json
from dataclasses import asdict

from cgraph import ADMG, CausalQuery
from cgraph.cli import main


def test_identification_cli_and_errors(tmp_path, capsys):
    graph = tmp_path / "graph.json"
    query = tmp_path / "query.json"
    output = tmp_path / "result.json"
    graph.write_text(json.dumps(ADMG(("t", "y"), (("t", "y"),), (("t", "y"),)).to_dict()))
    query.write_text(json.dumps(asdict(CausalQuery.interventional(treatment="t", outcome="y"))))
    assert (
        main(["identify", "--graph", str(graph), "--query", str(query), "--out", str(output)]) == 0
    )
    assert json.loads(output.read_text())["status"] == "not_identified"
    graph.write_text('{"kind":"dag","kind":"admg"}')
    assert main(["identify", "--graph", str(graph), "--query", str(query)]) == 2
    assert "Duplicate" in capsys.readouterr().err


def test_full_csv_cli_report_and_policy(tmp_path, capsys):
    import pathlib

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(441)
    experience = rng.normal(size=300)
    treatment = rng.binomial(1, 0.5, size=300)
    frame = pd.DataFrame(
        {
            "experience": experience,
            "training": treatment,
            "productivity": 2 * treatment + experience + rng.normal(size=300),
        }
    )
    data = tmp_path / "cohort.csv"
    frame.to_csv(data, index=False)
    configuration = pathlib.Path(__file__).resolve().parents[1] / "examples/audit-config.json"
    output = tmp_path / "audit"
    assert (
        main(["audit", "--config", str(configuration), "--data", str(data), "--out", str(output)])
        == 0
    )
    assert (output / "audit.json").exists()
    assert main(["policy", str(output / "audit.json")]) == 4
    assert "review_required" in capsys.readouterr().out
    assert main(["report", str(output / "audit.json"), "--html", str(output / "copy.html")]) == 0
    assert (output / "copy.html").read_text() == (output / "audit.html").read_text()
