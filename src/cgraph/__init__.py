"""Causal auditing; statistical dependencies are loaded only when requested."""

from .graph import ADMG, DAG, find_adjustment, latent_projection, validate_adjustment
from .query import CausalQuery
from .reports import AuditReport, CheckResult, Policy
from .symbolic import identify, identify_conditional, identify_family

__version__ = "0.1.0"
__all__ = [
    "ADMG",
    "DAG",
    "CausalQuery",
    "AuditReport",
    "CheckResult",
    "Policy",
    "identify",
    "identify_conditional",
    "identify_family",
    "find_adjustment",
    "latent_projection",
    "validate_adjustment",
    "CausalAudit",
    "AuditConfig",
]


def __getattr__(name):
    """Load optional numerical dependencies only for statistical entry points."""
    if name in ("CausalAudit", "AuditConfig"):
        try:
            from .audit import AuditConfig, CausalAudit
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Statistical auditing requires pip install 'cgraph-audit[stats]'"
            ) from exc
        return {"CausalAudit": CausalAudit, "AuditConfig": AuditConfig}[name]
    raise AttributeError(name)
