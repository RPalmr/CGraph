"""Repository source-style invariants and public package boundaries."""

import pathlib
import tokenize


def test_python_sources_use_docstrings_without_comment_tokens():
    root = pathlib.Path(__file__).resolve().parents[1]
    for directory in ("src", "tests", "examples", "benchmarks", "scripts"):
        for path in (root / directory).rglob("*.py"):
            with path.open("rb") as stream:
                assert not any(
                    token.type == tokenize.COMMENT for token in tokenize.tokenize(stream.readline)
                ), path


def test_package_exports_only_three_operations_and_supporting_types():
    import inspect

    import cgraph

    operations = {name for name in cgraph.__all__ if inspect.isfunction(getattr(cgraph, name))}
    assert operations == {"factorize", "intervene", "identify"}
    assert not hasattr(cgraph, "CausalAudit")
    assert not hasattr(cgraph, "Policy")
    assert not hasattr(cgraph, "identify_conditional")
