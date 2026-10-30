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
