"""Sensitivity analyses with explicit effect-scale and matched-design contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskRatio:
    """Adjusted conditional risk ratio with explicit outcome, contrast, and scope."""

    value: float
    confidence_interval: tuple[float, float] | None = None
    confidence_level: float = 0.95
    adjustment: tuple[str, ...] = ()
    outcome_event: str = ""
    contrast: str = ""
    scope: str = "conditional"

    def __post_init__(self):
        if not math.isfinite(self.value) or self.value <= 0 or not 0 < self.confidence_level < 1:
            raise ValueError("A positive finite risk ratio and valid confidence level are required")
        if self.scope != "conditional":
            raise ValueError("Only the conditional-RR E-value contract is supported")
        if not self.outcome_event or not self.contrast:
            raise ValueError("Declare the outcome event and treatment contrast")
        object.__setattr__(self, "adjustment", tuple(self.adjustment))
        if self.confidence_interval is not None:
            lo, hi = self.confidence_interval
            if not all(math.isfinite(v) and v > 0 for v in (lo, hi)) or not lo <= self.value <= hi:
                raise ValueError(
                    "Confidence interval must be positive, ordered, and contain the estimate"
                )
            object.__setattr__(self, "confidence_interval", (lo, hi))


def evalue(effect: RiskRatio) -> dict:
    """Compute conditional-RR point and confidence-limit E-values, without scale conversion."""
    if not isinstance(effect, RiskRatio):
        raise TypeError(
            "E-value requires a validated RiskRatio, not a mean difference or bare number"
        )

    def transform(rr):
        rr = max(rr, 1 / rr)
        return rr + math.sqrt(rr) * math.sqrt(rr - 1)

    interval = None
    if effect.confidence_interval:
        lo, hi = effect.confidence_interval
        interval = 1.0 if lo <= 1 <= hi else transform(lo if lo > 1 else hi)
    return {
        "point": transform(effect.value),
        "confidence_limit": interval,
        "scope": effect.scope,
        "scale": "risk_ratio",
        "interpretation": "Confounder associations with treatment and outcome needed to explain the conditional risk ratio under the E-value sensitivity model; not a probability of confounding",
    }


def rosenbaum_sign(differences, *, gammas=(1.0, 1.5, 2.0, 3.0), alternative="greater", alpha=0.05):
    """Bound one-sided paired sign-test p-values under Rosenbaum's assignment-odds model.

    Inputs are treated-minus-control differences from independent matched pairs.
    Exact zero differences are excluded. Bounds concern the sharp no-effect null,
    not an ATE confidence interval. Pair construction and match quality are caller
    responsibilities and cannot be verified from these differences alone.
    """
    import numpy as np
    from scipy.stats import binom

    d = np.asarray(differences, dtype=float)
    gammas = tuple(gammas)
    if d.ndim != 1 or not np.isfinite(d).all() or not gammas:
        raise ValueError("Finite one-dimensional differences and a Gamma grid are required")
    if any(not math.isfinite(g) or g < 1 for g in gammas) or list(gammas) != sorted(set(gammas)):
        raise ValueError("Gamma values must be unique, increasing, finite, and at least one")
    if alternative not in ("greater", "less") or not 0 < alpha < 1:
        raise ValueError("Specify a one-sided alternative and valid alpha")
    nonzero = d[d != 0]
    n = len(nonzero)
    if n == 0:
        raise ValueError("No nonzero pair differences")
    k = int(np.sum(nonzero > 0 if alternative == "greater" else nonzero < 0))
    bounds = [
        {
            "gamma": float(g),
            "lower_p_value": float(binom.sf(k - 1, n, 1 / (1 + g))),
            "upper_p_value": float(binom.sf(k - 1, n, g / (1 + g))),
        }
        for g in gammas
    ]
    crossing = next((b["gamma"] for b in bounds if b["upper_p_value"] >= alpha), None)
    return {
        "n_pairs": n,
        "zero_pairs_excluded": int(len(d) - n),
        "positive_in_alternative": k,
        "alternative": alternative,
        "alpha": alpha,
        "bounds": bounds,
        "first_grid_crossing": crossing,
        "threshold_exceeds_grid": crossing is None,
        "assumptions": "Independent matched pairs; bounded within-pair treatment odds; sharp no-effect null",
    }


def rosenbaum_pairs(outcome, treatment, pair_ids, **kwargs):
    """Validate matched pairs with one binary-treated unit each, then compute sign-test bounds."""
    import numpy as np

    y, t, ids = np.asarray(outcome, dtype=float), np.asarray(treatment), np.asarray(pair_ids)
    if y.ndim != 1 or y.shape != t.shape or y.shape != ids.shape or not np.isfinite(y).all():
        raise ValueError("Pair inputs must be aligned finite vectors")
    if not np.isin(t, [0, 1]).all():
        raise ValueError("Treatment must be binary 0/1")
    groups = {}
    for i, value in enumerate(ids.tolist()):
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError("Pair IDs cannot be missing")
        groups.setdefault(value, []).append(i)
    differences = []
    for indices in groups.values():
        if len(indices) != 2 or int(t[indices].sum()) != 1:
            raise ValueError("Each pair must contain one treated and one control unit")
        a, b = indices
        differences.append(float((y[a] - y[b]) * (2 * t[a] - 1)))
    return rosenbaum_sign(differences, **kwargs)
