"""Calibrated-domain conditional independence tests and fixed-family Holm adjustment."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .graph import DAG
from .reports import CheckResult


@dataclass(frozen=True)
class GaussianPartialCorrelation:
    """Fisher-z partial correlation, requiring a declared jointly Gaussian subvector."""

    def test(self, data, x, y, given, schema):
        names = [x, y, *given]
        if any(schema.get(n) != "continuous_gaussian" for n in names):
            return {
                "status": "unsupported",
                "reason": "Requires continuous_gaussian schema for all test variables",
            }
        try:
            values = data[names].to_numpy(dtype=float)
        except (ValueError, TypeError):
            return {"status": "unsupported", "reason": "Gaussian test requires numeric inputs"}
        if not np.isfinite(values).all():
            return {"status": "insufficient_data", "reason": "Missing or nonfinite test inputs"}
        n, k = len(values), len(given)
        if n <= k + 3:
            return {
                "status": "insufficient_data",
                "reason": "Insufficient residual degrees of freedom",
            }
        design = np.column_stack((np.ones(n), values[:, 2:]))
        if np.linalg.matrix_rank(design) != k + 1:
            return {"status": "insufficient_data", "reason": "Rank-deficient conditioning set"}
        residual = values[:, :2] - design @ np.linalg.lstsq(design, values[:, :2], rcond=None)[0]
        norms = np.linalg.norm(residual, axis=0)
        if np.any(norms < np.finfo(float).eps * max(1, np.linalg.norm(values))):
            return {"status": "insufficient_data", "reason": "Degenerate residual variation"}
        r = float(np.dot(residual[:, 0], residual[:, 1]) / np.prod(norms))
        r = np.clip(r, -1 + 1e-15, 1 - 1e-15)
        z = float(np.arctanh(r) * np.sqrt(n - k - 3))
        return {
            "status": "completed",
            "p_value": float(2 * stats.norm.sf(abs(z))),
            "statistic": z,
            "partial_correlation": float(r),
            "n": n,
            "assumptions": "IID jointly Gaussian observations; Fisher-z asymptotic calibration",
        }


@dataclass(frozen=True)
class DiscreteConditionalIndependence:
    """Stratified likelihood-ratio independence test with conservative cell-count checks."""

    min_expected: float = 5.0

    def test(self, data, x, y, given, schema):
        names = [x, y, *given]
        if any(schema.get(n) not in ("binary", "categorical") for n in names):
            return {"status": "unsupported", "reason": "Requires declared discrete variables"}
        if self.min_expected < 1:
            raise ValueError("min_expected must be at least one")
        if data[names].isna().any().any():
            return {"status": "insufficient_data", "reason": "Missing test inputs"}
        groups = data.groupby(list(given), observed=True, sort=False) if given else [((), data)]
        statistic, df, strata = 0.0, 0, 0
        for _, group in groups:
            table = pd.crosstab(group[x], group[y]).to_numpy()
            if min(table.shape) < 2:
                return {
                    "status": "insufficient_data",
                    "reason": "A conditioning stratum lacks variation",
                }
            result = stats.chi2_contingency(table, correction=False, lambda_="log-likelihood")
            if (result.expected_freq < self.min_expected).any():
                return {
                    "status": "insufficient_data",
                    "reason": "Sparse expected contingency cells",
                }
            statistic += result.statistic
            df += result.dof
            strata += 1
        return {
            "status": "completed",
            "p_value": float(stats.chi2.sf(statistic, df)),
            "statistic": float(statistic),
            "df": int(df),
            "strata": strata,
            "n": len(data),
            "assumptions": "IID discrete observations with adequate conditional cell counts",
        }


def holm(pvalues, *, family_size=None):
    """Adjust valid p-values; untested planned hypotheses count conservatively as p=1."""
    p = np.asarray(pvalues, dtype=float)
    size = len(p) if family_size is None else family_size
    if size < len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values or family size")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate((size - np.arange(len(p))) * p[order]))
    return adjusted.tolist()


def dag_implications(graph: DAG):
    """Enumerate unique pairwise local-Markov implications for a fixed DAG."""
    if not isinstance(graph, DAG):
        raise ValueError("Local-Markov implication generation requires a DAG")
    seen = set()
    for node in graph.nodes:
        excluded = graph.descendants((node,)) | graph.parents[node]
        for other in sorted(set(graph.nodes) - excluded):
            x, y = sorted((node, other))
            key = (x, y, tuple(sorted(graph.parents[node])))
            if key not in seen:
                seen.add(key)
                yield key


def audit_implications(graph, data, schema, *, test=None, alpha=0.05, max_tests=200):
    """Audit a prespecified DAG family; keep unsupported and unexecuted tests visible."""
    if not 0 < alpha < 1 or max_tests < 1:
        raise ValueError("Invalid test budget or alpha")
    if not isinstance(graph, DAG):
        return (
            CheckResult(
                "graph_implications",
                "unsupported",
                "statistical_test",
                "Automatic implication generation is currently DAG-only",
            ),
        )
    test = test or GaussianPartialCorrelation()
    planned = list(itertools.islice(dag_implications(graph), max_tests + 1))
    truncated = len(planned) > max_tests
    planned = planned[:max_tests]
    records, complete = [], []
    for i, (x, y, z) in enumerate(planned):
        if not set((x, y, *z)) <= set(data.columns):
            result = {"status": "insufficient_data", "reason": "Required variable is absent"}
        else:
            result = test.test(data, x, y, z, schema)
        records.append((x, y, z, result))
        if result["status"] == "completed":
            complete.append(i)
    adjusted = holm([records[i][3]["p_value"] for i in complete], family_size=len(planned))
    adjusted_by_index = dict(zip(complete, adjusted, strict=True))
    checks = []
    for i, (x, y, z, result) in enumerate(records):
        status = result["status"]
        metrics = dict(result) | {
            "x": x,
            "y": y,
            "given": list(z),
            "alpha": alpha,
            "family": "graph_implications",
            "family_size": len(planned),
        }
        if status == "completed":
            p = adjusted_by_index[i]
            metrics["adjusted_p_value"] = p
            finding = "violation_detected" if p < alpha else "no_violation_detected"
            summary = f"Test of {x} independent of {y} given {list(z)}"
        else:
            finding, summary = None, result["reason"]
        checks.append(CheckResult(f"ci_{i}", status, "statistical_test", summary, finding, metrics))
    coverage = {
        "planned": len(planned),
        "completed": len(complete),
        "truncated": truncated,
        "scope": "Fixed graph-derived prefix; pairwise implications only",
    }
    checks.append(
        CheckResult(
            "implication_coverage",
            "resource_limited" if truncated else "completed",
            "statistical_test",
            "Graph implication test coverage; non-rejection is not graph validation",
            None if truncated else "informational" if complete else "inconclusive",
            coverage,
        )
    )
    return tuple(checks)
