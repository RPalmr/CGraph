"""Three-operation CLI and strict JSON parsing."""

import json

from cgraph import ADMG, DAG
from cgraph.cli import main


def test_three_commands(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(DAG(("t", "y"), (("t", "y"),)).to_dict()))
    out = tmp_path / "result.json"
    for command, options in (
        ("factorize", []),
        ("intervene", ["--values", '{"t":1}']),
        ("identify", ["--treatment", "t", "--outcome", "y"]),
    ):
        assert main([command, "--graph", str(path), "--out", str(out), *options]) == 0
        assert json.loads(out.read_text())["operation"] == command
    path.write_text(json.dumps(ADMG(("t", "y"), (("t", "y"),), (("t", "y"),)).to_dict()))
    assert (
        main(
            [
                "identify",
                "--graph",
                str(path),
                "--treatment",
                "t",
                "--outcome",
                "y",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["status"] == "not_identified"
    assert main(["factorize", "--graph", str(path)]) == 2
    path.write_text('{"kind":"dag","kind":"admg"}')
    assert main(["factorize", "--graph", str(path)]) == 2


def test_data_first_discovery_cli(tmp_path):
    import numpy as np

    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(2, 5000))
    c = a + b + rng.normal(size=5000)
    data = tmp_path / "cohort.csv"
    np.savetxt(data, np.column_stack((a, b, c)), delimiter=",", header="A,B,C", comments="")
    output = tmp_path / "discovered.json"
    dot = tmp_path / "discovered.dot"
    assert (
        main(
            [
                "discover",
                "--data",
                str(data),
                "--out",
                str(output),
                "--dot",
                str(dot),
                "--timeout",
                "60",
            ]
        )
        == 0
    )
    value = json.loads(output.read_text())
    assert value["graph"]["kind"] == "cpdag"
    assert value["status"] == "completed"
    assert dot.read_text().startswith("digraph CGraph")
