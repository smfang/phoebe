"""
Sara domain modules — pluggable safety verification per problem domain.

Each domain (DAO function calls, medical triage, code execution, etc.) packages
its own validator/simulator/judge stages plus a risk-tier function and a
tier→enforcement-mode mapping. The generic `DomainEnforcement` orchestrator
runs those stages and produces a uniform `EnforcementDecision`.

Other modules trained to different safety standards can be swapped in by
registering them with the `DomainRegistry` at bootstrap.
"""

from src.domains.base import (
    Action,
    DomainModule,
    EnforcementDecision,
    EnforcementMode,
    EvalRequest,
    EvalState,
    RiskTier,
    Stage,
    StageResult,
)
from src.domains.enforcement import DomainEnforcement
from src.domains.registry import DomainRegistry, UnknownDomainError

__all__ = [
    "Action",
    "DomainEnforcement",
    "DomainModule",
    "DomainRegistry",
    "EnforcementDecision",
    "EnforcementMode",
    "EvalRequest",
    "EvalState",
    "RiskTier",
    "Stage",
    "StageResult",
    "UnknownDomainError",
]
