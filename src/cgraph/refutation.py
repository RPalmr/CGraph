"""Reproducible estimator stress tests with explicit synthetic-task interpretation."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

from .reports import CheckResult


def stress_tests(
    estimator,
    X,
    y,
    *,
    treatment,
    control=0,
    treated=1,
    placebo_repetitions=0,
    subset_repetitions=0,
    subset_fraction=0.8,
    random_state=0,
):
    """Refit cloned pipelines on synthetic placebos and overlapping random subsets.

    Placebos replace treatment with an independent Bernoulli variable and measure
    behavior in that synthetic task. They are not randomization tests for the
    original observational treatment. Subset ranges describe stability and are
    not independent-sample confidence intervals.
    """
    if (
        any(
            not isinstance(n, int) or not 0 <= n <= 1000
            for n in (placebo_repetitions, subset_repetitions)
        )
        or not 0 < subset_fraction < 1
    ):
        raise ValueError("Invalid resampling configuration")
    y = np.asarray(y)
    checks = []
    for kind, repetitions, stream_id in (
        ("synthetic_placebo", placebo_repetitions, 17),
        ("subset_stability", subset_repetitions, 29),
    ):
        if not repetitions:
            continue
        estimates, failures = [], []
        seeds = np.random.SeedSequence([random_state, stream_id]).spawn(repetitions)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            candidate = clone(estimator)
            candidate.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
            if kind == "synthetic_placebo":
                frame = X.copy()
                frame[treatment] = np.where(rng.random(len(X)) < 0.5, treated, control)
                outcomes = y
            else:
                indices = rng.choice(len(X), max(4, int(len(X) * subset_fraction)), replace=False)
                frame, outcomes = X.iloc[indices], y[indices]
            try:
                candidate.fit(
                    frame, outcomes, treatment=treatment, control=control, treated=treated
                )
                estimates.append(candidate.estimate_.value)
            except (ValueError, FloatingPointError) as exc:
                failures.append(str(exc))
        metrics = {
            "requested": repetitions,
            "completed": len(estimates),
            "failures": failures,
            "interpretation": "Synthetic pipeline diagnostic, not evidence of no confounding"
            if kind == "synthetic_placebo"
            else "Stability across overlapping subsets; differences may reflect heterogeneity",
        }
        if estimates:
            metrics.update(
                {
                    "mean": float(np.mean(estimates)),
                    "standard_deviation": float(np.std(estimates, ddof=1))
                    if len(estimates) > 1
                    else None,
                    "quantiles": np.quantile(estimates, [0, 0.05, 0.5, 0.95, 1]).tolist(),
                }
            )
        checks.append(
            CheckResult(
                kind,
                "completed" if estimates else "insufficient_data",
                "stress_scenario",
                metrics["interpretation"],
                ("inconclusive" if failures else "informational") if estimates else None,
                metrics,
            )
        )
    return tuple(checks)
