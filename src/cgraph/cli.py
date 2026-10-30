"""Local-only command line interface for identification, audits, reports, and policies."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .graph import ADMG
from .query import CausalQuery
from .reports import AuditReport, Policy, atomic_write
from .symbolic import identify, identify_conditional


def _load(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject, object_pairs_hook=unique)


def _emit(value, path=None):
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path:
        atomic_write(path, content)
    else:
        print(content, end="")


def _audit(args):
    import pandas as pd

    from .audit import AuditConfig, CausalAudit
    from .estimation import AIPW
    from .testing import DiscreteConditionalIndependence, GaussianPartialCorrelation

    configuration = _load(args.config)
    allowed = {
        "graph",
        "query",
        "schema",
        "assumptions",
        "random_state",
        "audit",
        "estimator",
        "dataset_id",
    }
    if not isinstance(configuration, dict) or set(configuration) - allowed:
        raise ValueError("Unknown audit configuration keys")
    graph = ADMG.from_dict(configuration["graph"])
    query = CausalQuery(**configuration["query"])
    options = dict(configuration.get("audit", {}))
    method = options.pop("ci_test", "gaussian")
    if method not in ("gaussian", "discrete"):
        raise ValueError("ci_test must be gaussian or discrete")
    options["ci_test"] = (
        GaussianPartialCorrelation() if method == "gaussian" else DiscreteConditionalIndependence()
    )
    config = AuditConfig(**options)
    settings = configuration.get("estimator")
    estimator = None
    if settings is not None:
        if set(settings) - {
            "adjustment",
            "cv",
            "confidence_level",
            "overlap_threshold",
            "categorical",
        }:
            raise ValueError("Unsupported estimator configuration")
        estimator = AIPW(**settings, random_state=configuration.get("random_state", 0))
    if Path(args.data).suffix.lower() != ".csv":
        raise ValueError("CLI data input currently supports CSV files")
    frame = pd.read_csv(args.data)
    if len(query.outcome) != 1 or query.outcome[0] not in frame:
        raise ValueError("Dataset must contain the declared scalar outcome")
    y = frame.pop(query.outcome[0])
    model = CausalAudit(graph, query, estimator, config, configuration.get("random_state", 0)).fit(
        frame,
        y,
        schema=configuration.get("schema"),
        assumptions=configuration.get("assumptions", ()),
        dataset_id=configuration.get("dataset_id", "caller_csv"),
    )
    report = model.audit_report()
    destination = Path(args.out)
    report.to_json(destination / "audit.json")
    report.to_html(destination / "audit.html")
    _emit({"report": str(destination / "audit.json"), "policy": report.evaluate().to_dict()})
    return 0


def main(argv=None):
    """Run a command; computation success is separate from deployment permission."""
    parser = argparse.ArgumentParser(
        prog="cgraph", description="Causal graph auditing under explicit assumptions"
    )
    parser.add_argument("--version", action="version", version="cgraph-audit 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    identification = commands.add_parser(
        "identify", help="Identify a joint intervention symbolically"
    )
    identification.add_argument("--graph", required=True)
    identification.add_argument("--query", required=True)
    identification.add_argument("--out")
    identification.add_argument("--given", nargs="*", default=[])
    identification.add_argument("--timeout", type=float, default=10.0)
    identification.add_argument("--max-calls", type=int, default=10000)
    audit = commands.add_parser("audit", help="Audit a CSV dataset using a JSON configuration")
    audit.add_argument("--config", required=True)
    audit.add_argument("--data", required=True)
    audit.add_argument("--out", required=True)
    report = commands.add_parser("report", help="Render an existing report to standalone HTML")
    report.add_argument("input")
    report.add_argument("--html", required=True)
    policy = commands.add_parser("policy", help="Enforce an explicit deployment policy")
    policy.add_argument("input")
    policy.add_argument("--policy")
    args = parser.parse_args(argv)
    try:
        if args.command == "identify":
            graph = ADMG.from_dict(_load(args.graph))
            query = CausalQuery(**_load(args.query))
            result = (
                identify_conditional(
                    graph, query, args.given, timeout=args.timeout, max_calls=args.max_calls
                )
                if args.given
                else identify(graph, query, timeout=args.timeout, max_calls=args.max_calls)
            )
            _emit(
                {
                    "graph": graph.to_dict(),
                    "query": asdict(query),
                    "given": args.given,
                    **result.to_dict(),
                },
                args.out,
            )
            return 4 if result.status == "resource_limited" else 0
        if args.command == "audit":
            return _audit(args)
        loaded = AuditReport.from_json(Path(args.input).read_text(encoding="utf-8"))
        if args.command == "report":
            loaded.to_html(args.html)
            return 0
        decision = loaded.evaluate(Policy(**_load(args.policy)) if args.policy else Policy())
        _emit(decision.to_dict())
        return {"allow": 0, "review_required": 4, "block": 3}[decision.decision]
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        print(f"cgraph: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"cgraph: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
