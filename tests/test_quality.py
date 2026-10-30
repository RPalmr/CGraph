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
