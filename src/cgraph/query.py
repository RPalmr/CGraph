"""Explicit causal queries and numerical treatment contrasts."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CausalQuery:
    """A joint intervention query, optionally with a binary-treatment ATE contrast."""

    treatment: tuple[str, ...]
    outcome: tuple[str, ...]
    kind: str = "interventional"
    control: float = 0
    treated: float = 1
    target_population: str = "unspecified"

    def __post_init__(self):
        for field in ("treatment", "outcome"):
            values = getattr(self, field)
            values = (values,) if isinstance(values, str) else tuple(values)
            if (
                not values
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"Invalid {field} variables")
            object.__setattr__(self, field, tuple(sorted(values)))
        if set(self.treatment) & set(self.outcome):
            raise ValueError("Treatment and outcome must be disjoint")
        if not isinstance(self.target_population, str) or not self.target_population:
            raise ValueError("Target population must be a nonempty description")
        if any(
            not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (self.control, self.treated)
        ):
            raise ValueError("Treatment contrast levels must be finite numeric values")
        if self.kind not in ("interventional", "ate"):
            raise ValueError("Unsupported query kind")
        if self.kind == "ate" and (
            len(self.treatment) != 1 or len(self.outcome) != 1 or self.control == self.treated
        ):
            raise ValueError("ATE requires distinct levels and one treatment/outcome")

    @classmethod
    def interventional(cls, *, treatment, outcome):
        """Construct a symbolic joint-intervention query."""
        return cls(treatment, outcome)

    @classmethod
    def ate(cls, *, treatment, outcome, control=0, treated=1, target_population="unspecified"):
        """Construct an average mean-difference query with explicit treatment levels."""
        return cls(treatment, outcome, "ate", control, treated, target_population)
