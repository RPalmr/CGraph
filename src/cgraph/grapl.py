"""Native GRAPL text export for explicit DAGs and acyclic directed mixed graphs.

The grammar is documented by the independently maintained GRAPL project:
https://github.com/max-little/GRAPL. It permits node declarations, directed
arrows and bidirected arrows; it has no partial-orientation endpoint notation.
This module implements output from that public grammar without importing or
copying the reference parser.
"""

import re
from collections.abc import Mapping

from ._serialization import atomic_write
from .graph import ADMG

_IDENTIFIER = re.compile(r"[A-Za-z_]+[0-9]*", re.ASCII)


def to_grapl(graph, path=None, *, title="CGraph", node_names=None):
    """Serialize a DAG/ADMG to native GRAPL, optionally writing it atomically.

    Names must follow GRAPL's identifier grammar: ASCII letters or underscores,
    followed by optional trailing digits. An explicit one-to-one node_names
    mapping can replace incompatible variable labels. Callers should retain
    that mapping with their dataset metadata. No edge is dropped or reoriented.
    """
    if not isinstance(graph, ADMG):
        raise ValueError("Native GRAPL export requires an explicit DAG or ADMG")
    if not isinstance(title, str) or any(c == '"' or ord(c) < 32 or ord(c) == 127 for c in title):
        raise ValueError(
            "GRAPL titles must be single-line text without quotes or control characters"
        )
    if not graph.nodes:
        raise ValueError("The native GRAPL parser requires at least one node declaration")
    if node_names is None:
        names = {name: name for name in graph.nodes}
    elif not isinstance(node_names, Mapping) or set(node_names) != set(graph.nodes):
        raise ValueError("node_names must map every graph node exactly once")
    else:
        names = dict(node_names)
    if any(
        not isinstance(name, str) or _IDENTIFIER.fullmatch(name) is None for name in names.values()
    ):
        raise ValueError(
            "GRAPL names require letters/underscores followed by optional trailing digits; provide node_names for other labels"
        )
    if len(set(names.values())) != len(names):
        raise ValueError("GRAPL node names must be unique")
    lines = [f'"{title}";']
    lines.extend(f"{names[n]};" for n in graph.nodes)
    lines.extend(f"{names[u]} -> {names[v]};" for u, v in graph.directed)
    lines.extend(f"{names[u]} <-> {names[v]};" for u, v in graph.bidirected)
    content = "\n".join(lines) + "\n"
    if path is not None:
        atomic_write(path, content)
    return content
