"""Command line interface exposing only factorize, intervene, and identify."""

import argparse
import json
import sys
from pathlib import Path

from ._serialization import atomic_write
from .distributions import factorize, intervene
from .graph import ADMG
from .symbolic import identify


def _parse(content):
    def reject(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(content, parse_constant=reject, object_pairs_hook=unique)


def main(argv=None):
    """Emit mathematical results; computational failure is distinct from nonidentification."""
    parser = argparse.ArgumentParser(prog="cgraph", description=__doc__)
    parser.add_argument("--version", action="version", version="cgraph-audit 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("factorize", "intervene", "identify"):
        command = commands.add_parser(name)
        command.add_argument("--graph", required=True)
        command.add_argument("--out")
        if name == "intervene":
            command.add_argument("--values", required=True, help='JSON assignments, e.g. {"T":1}')
        elif name == "identify":
            command.add_argument("--treatment", nargs="+", required=True)
            command.add_argument("--outcome", nargs="+", required=True)
            command.add_argument("--method", choices=("auto", "id"), default="auto")
            command.add_argument("--timeout", type=float, default=10.0)
            command.add_argument("--max-calls", type=int, default=10000)
            command.add_argument("--max-sets", type=int, default=10000)
    args = parser.parse_args(argv)
    try:
        graph = ADMG.from_dict(_parse(Path(args.graph).read_text(encoding="utf-8")))
        code = 0
        if args.command == "factorize":
            expression = factorize(graph)
            result = {"formula": str(expression), "expression": expression.to_dict()}
        elif args.command == "intervene":
            result = intervene(graph, _parse(args.values)).to_dict()
        else:
            identification = identify(
                graph,
                treatment=args.treatment,
                outcome=args.outcome,
                method=args.method,
                timeout=args.timeout,
                max_calls=args.max_calls,
                max_sets=args.max_sets,
            )
            result = identification.to_dict()
            code = 4 if identification.status == "resource_limited" else 0
        result = {"operation": args.command, "graph": graph.to_dict(), **result}
        if args.command == "identify":
            result.update({"treatment": args.treatment, "outcome": args.outcome})
        content = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.out:
            atomic_write(args.out, content)
        else:
            print(content, end="")
        return code
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"cgraph: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"cgraph: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
