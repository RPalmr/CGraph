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
