"""Immutable audit evidence, strict JSON persistence, and explicit deployment policies."""

from __future__ import annotations

import html
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

EXECUTION = frozenset(
    ("completed", "not_applicable", "unsupported", "insufficient_data", "resource_limited", "error")
)
FINDINGS = frozenset(
    ("violation_detected", "no_violation_detected", "inconclusive", "informational")
)
EVIDENCE = frozenset(
    (
        "graph_theorem",
        "statistical_test",
        "estimator_diagnostic",
        "sensitivity_analysis",
        "stress_scenario",
        "user_assertion",
    )
)


def freeze(value):
    """Copy JSON-compatible data into recursively immutable containers."""
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("JSON keys must be strings")
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Not a finite JSON value: {type(value).__name__}")


def thaw(value):
    """Produce an independent JSON-compatible snapshot."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


def atomic_write(path, content: str):
    """Write an artifact atomically in its destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".cgraph-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class CheckResult:
    """Scientific finding and execution status are independent fields."""

    check_id: str
    execution_status: str
    evidence_type: str
    summary: str
    finding: str | None = None
    metrics: Mapping = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.check_id or not isinstance(self.summary, str):
            raise ValueError("Checks need an identifier and summary")
        if self.execution_status not in EXECUTION or self.evidence_type not in EVIDENCE:
            raise ValueError("Invalid evidence or execution status")
        if self.execution_status == "completed":
            if self.finding not in FINDINGS:
                raise ValueError("Completed checks require a valid finding")
        elif self.finding is not None:
            raise ValueError("Unexecuted checks cannot have a scientific finding")
        object.__setattr__(self, "metrics", freeze(self.metrics))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))

    def to_dict(self):
        """Return a detached record ready for JSON serialization."""
        return {
            "check_id": self.check_id,
            "execution_status": self.execution_status,
            "evidence_type": self.evidence_type,
            "summary": self.summary,
            "finding": self.finding,
            "metrics": thaw(self.metrics),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class Policy:
    """Deployment policy; allowing an audit is not a causal-validity certificate."""

    require_estimate: bool = True
    require_empirical: bool = True
    require_declared_assumptions: bool = True
    version: str = "1"


@dataclass(frozen=True)
class PolicyDecision:
    """Policy action and the evidence that triggered it."""

    decision: str
    reasons: tuple[str, ...]
    policy_version: str

    def to_dict(self):
        """Return a machine-readable policy decision."""
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class AuditReport:
    """Versioned report with immutable nested evidence and no raw rows by default."""

    graph: Mapping
    query: Mapping
    identification: Mapping
    checks: tuple[CheckResult, ...] = ()
    estimate: Mapping | None = None
    provenance: Mapping = field(default_factory=dict)
    schema_version: str = "1"

    def __post_init__(self):
        if self.schema_version != "1":
            raise ValueError("Unsupported report schema version")
        for name in ("graph", "query", "identification", "estimate", "provenance"):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        object.__setattr__(self, "checks", tuple(self.checks))
        if any(not isinstance(c, CheckResult) for c in self.checks):
            raise ValueError("Invalid check record")
        ids = [c.check_id for c in self.checks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate check identifiers")
        if self.identification.get("status") not in (
            "identified",
            "not_identified",
            "unsupported",
            "resource_limited",
            "invalid_input",
        ):
            raise ValueError("Invalid identification status")

    def to_dict(self):
        """Return a detached report snapshot."""
        return {
            "schema_version": self.schema_version,
            "graph": thaw(self.graph),
            "query": thaw(self.query),
            "identification": thaw(self.identification),
            "checks": [c.to_dict() for c in self.checks],
            "estimate": thaw(self.estimate),
            "provenance": thaw(self.provenance),
        }

    def to_json(self, path=None) -> str:
        """Serialize strictly, optionally replacing a local artifact atomically."""
        result = json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        if path is not None:
            atomic_write(path, result)
        return result

    @classmethod
    def from_json(cls, content: str):
        """Load a report from JSON text; reject unknown fields and nonfinite numbers."""

        def reject_constant(value):
            raise ValueError(f"Invalid JSON constant: {value}")

        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result

        value = json.loads(content, parse_constant=reject_constant, object_pairs_hook=unique_pairs)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "graph",
            "query",
            "identification",
            "checks",
            "estimate",
            "provenance",
        }:
            raise ValueError("Invalid report schema")
        try:
            value["checks"] = tuple(CheckResult(**c) for c in value["checks"])
            return cls(**value)
        except (TypeError, KeyError) as exc:
            raise ValueError("Invalid report record") from exc

    def evaluate(self, policy: Policy | None = None) -> PolicyDecision:
        """Apply explicit gates without changing or averaging scientific evidence."""
        policy = policy or Policy()
        block, review = [], []
        status = self.identification["status"]
        if status in ("not_identified", "invalid_input"):
            block.append(f"Identification: {status}")
        elif status != "identified":
            review.append(f"Identification incomplete: {status}")
        if policy.require_estimate and self.estimate is None:
            block.append("Required causal estimate is unavailable")
        if policy.require_declared_assumptions and not self.provenance.get("assumptions"):
            review.append("Required assumptions have not been explicitly declared")
        empirical = [c for c in self.checks if c.evidence_type == "statistical_test"]
        if policy.require_empirical and not empirical:
            review.append("No empirical checks were available")
        for c in self.checks:
            if c.finding == "violation_detected":
                review.append(c.check_id + ": " + c.summary)
            elif (
                c.evidence_type == "statistical_test"
                and policy.require_empirical
                and (c.execution_status != "completed" or c.finding == "inconclusive")
            ):
                review.append(c.check_id + ": empirical evidence incomplete")
            elif c.execution_status == "error":
                review.append(c.check_id + ": check execution failed")
        decision = "block" if block else "review_required" if review else "allow"
        return PolicyDecision(decision, tuple(block + review), policy.version)

    def to_html(self, path=None) -> str:
        """Render an escaped, standalone report without scripts or external requests."""
        rows = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in (c.check_id, c.execution_status, c.finding or "—", c.summary)
            )
            + "</tr>"
            for c in self.checks
        )
        result = (
            '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CGraph audit</title><style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#183044}table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #ccd5dd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f8;padding:1rem}</style><h1>CGraph causal audit</h1><p>Identification is conditional on the supplied graph. Non-rejection does not establish causal validity.</p><table><thead><tr><th>Check</th><th>Execution</th><th>Finding</th><th>Explanation</th></tr></thead><tbody>'
            + rows
            + "</tbody></table><details><summary>Complete evidence and provenance</summary><pre>"
            + html.escape(self.to_json())
            + "</pre></details></html>"
        )
        if path is not None:
            atomic_write(path, result)
        return result

    def compare(self, other: AuditReport) -> dict:
        """List changed evidence sections without equating different causal targets."""
        a, b = self.to_dict(), other.to_dict()
        return {
            "changed_sections": [k for k in a if a[k] != b[k]],
            "same_query": a["query"] == b["query"],
            "same_graph": a["graph"] == b["graph"],
        }
