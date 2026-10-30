"""Native GRAPL compatibility and prevention of silent changes to causal semantics."""

import contextlib
import functools
import importlib
import io
from pathlib import Path

import pytest

from cgraph import ADMG, DAG, DiscoveryResult, PartialEdge, PartialGraph


@pytest.fixture
def collider():
    return PartialGraph(
        ("A", "B", "C"),
        (PartialEdge("A", "C", "tail", "arrow"), PartialEdge("B", "C", "tail", "arrow")),
        "cpdag",
    )


def test_native_dag_export_declares_nodes_before_edges_and_is_deterministic(tmp_path, collider):
    output = tmp_path / "graph.grapl"
    expected = '"Collider";\nA;\nB;\nC;\nA -> C;\nB -> C;\n'
    assert collider.to_grapl(output, title="Collider") == expected
    assert output.read_text() == expected
    assert collider.to_dag().to_grapl(title="Collider") == expected
    result = DiscoveryResult("completed", collider, "pc", ("Test assumptions",))
    assert result.to_grapl(title="Collider") == expected


def test_admg_export_preserves_bows_and_bidirected_edges():
    graph = ADMG(("X", "M", "Y", "Isolated"), (("X", "M"), ("M", "Y"), ("X", "Y")), (("X", "Y"),))
    text = graph.to_grapl()
    assert "Isolated;\n" in text
    assert "X -> Y;\n" in text
    assert text.count("X <-> Y;") == 1


@pytest.mark.parametrize(
    "kind,marks",
    [
        ("cpdag", ("tail", "tail")),
        ("pag", ("circle", "arrow")),
        ("pag", ("arrow", "arrow")),
        ("pag", ("tail", "arrow")),
        ("pdag", ("tail", "arrow")),
    ],
)
def test_native_export_rejects_partial_and_ancestral_graphs_without_touching_output(
    tmp_path, kind, marks
):
    graph = PartialGraph(("A", "B"), (PartialEdge("A", "B", *marks),), kind)
    path = tmp_path / "unchanged.grapl"
    path.write_text("previous valid artifact")
    with pytest.raises(ValueError, match="Native GRAPL"):
        graph.to_grapl(path)
    assert path.read_text() == "previous valid artifact"


def test_grapl_identifier_validation_and_explicit_mapping():
    graph = DAG(("Blood pressure", "HbA1c"), (("Blood pressure", "HbA1c"),))
    with pytest.raises(ValueError, match="node_names"):
        graph.to_grapl()
    text = graph.to_grapl(node_names={"Blood pressure": "Blood_pressure", "HbA1c": "HbA"})
    assert "Blood_pressure -> HbA;" in text
    for names in (
        {"Blood pressure": "A"},
        {"Blood pressure": "A", "HbA1c": "A"},
        {"Blood pressure": "A1B", "HbA1c": "B"},
    ):
        with pytest.raises(ValueError):
            graph.to_grapl(node_names=names)
    for title in ('Bad"title', "Bad\ntitle", "Bad\rtitle"):
        with pytest.raises(ValueError):
            graph.to_grapl(title=title)


def test_incomplete_discovery_and_empty_graph_have_no_native_export(collider):
    failed = DiscoveryResult("resource_limited", None, "pc", ())
    with pytest.raises(ValueError, match="completed"):
        failed.to_grapl()
    with pytest.raises(ValueError, match="node"):
        DAG(()).to_grapl()


@pytest.mark.filterwarnings("ignore:invalid escape sequence:SyntaxWarning")
def test_export_with_local_reference_grapl_parser(monkeypatch):
    reference = Path(__file__).resolve().parents[1] / "GRAPL"
    if not (reference / "grapl/dsl.py").exists():
        pytest.skip("Local GRAPL checkout is an optional interoperability oracle")
    monkeypatch.syspath_prepend(str(reference))
    dsl = importlib.import_module("grapl.dsl")
    monkeypatch.setattr(
        dsl.yacc, "yacc", functools.partial(dsl.yacc.yacc, write_tables=False, debug=False)
    )
    graph = ADMG(
        ("X1", "Y", "Mediator", "Isolated"), (("X1", "Mediator"), ("Mediator", "Y")), (("X1", "Y"),)
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        decoded = dsl.GraplDSL().readgrapl(graph.to_grapl(title="Round trip"))
    assert decoded is not None
    assert set(decoded.vars) == set(graph.nodes)
    assert {(parent, node) for node in decoded.vars for parent in decoded.pa({node})} == set(
        graph.directed
    )
    assert {
        tuple(sorted((node, other))) for node in decoded.vars for other in decoded.bi({node})
    } == set(graph.bidirected)


def test_cli_writes_native_grapl_and_rejects_partial_graphs(
    monkeypatch, tmp_path, collider, capsys
):
    import cgraph.cli as cli

    data = tmp_path / "input.csv"
    data.write_text("A,B,C\n1,2,3\n")
    complete = DiscoveryResult("completed", collider, "pc", ("Test assumptions",))
    monkeypatch.setattr(cli, "discover", lambda *args, **kwargs: complete)
    target = tmp_path / "result.grapl"
    assert cli.main(["discover", "--data", str(data), "--out", str(target)]) == 0
    assert target.read_text() == complete.to_grapl()
    json_path = tmp_path / "result.json"
    assert (
        cli.main(["discover", "--data", str(data), "--out", str(json_path), "--grapl", str(target)])
        == 0
    )
    assert DiscoveryResult.from_json(json_path.read_text()).graph == collider
    partial = PartialGraph(("A", "B"), (PartialEdge("A", "B", "tail", "tail"),), "cpdag")
    monkeypatch.setattr(
        cli, "discover", lambda *args, **kwargs: DiscoveryResult("completed", partial, "pc", ())
    )
    before = target.read_text()
    assert cli.main(["discover", "--data", str(data), "--out", str(target)]) == 2
    assert "Native GRAPL cannot represent" in capsys.readouterr().err
    assert target.read_text() == before


def test_cli_preserves_failure_status_without_creating_native_graph(monkeypatch, tmp_path, capsys):
    import cgraph.cli as cli

    data = tmp_path / "input.csv"
    data.write_text("A,B\n1,2\n")
    limited = DiscoveryResult("resource_limited", None, "pc", (), {"reason": "Test budget"})
    monkeypatch.setattr(cli, "discover", lambda *args, **kwargs: limited)
    target = tmp_path / "result.grapl"
    assert cli.main(["discover", "--data", str(data), "--out", str(target)]) == 4
    assert not target.exists()
    assert '"status": "resource_limited"' in capsys.readouterr().err


def test_cli_rejects_colliding_artifact_paths_before_discovery(monkeypatch, tmp_path, collider):
    import cgraph.cli as cli

    data = tmp_path / "input.csv"
    original = "A,B,C\n1,2,3\n"
    data.write_text(original)
    calls = []

    def discover(*args, **kwargs):
        calls.append(True)
        return DiscoveryResult("completed", collider, "pc", ())

    monkeypatch.setattr(cli, "discover", discover)
    output = str(tmp_path / "result.json")
    for options in (
        ["--out", output, "--grapl", output],
        ["--out", output, "--dot", output],
        ["--grapl", str(data)],
    ):
        assert cli.main(["discover", "--data", str(data), *options]) == 2
    assert not calls
    assert data.read_text() == original
