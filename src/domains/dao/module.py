"""Builds the DAO `DomainModule` ready to register with `DomainRegistry`."""

from __future__ import annotations

from typing import Any

from src.domains.base import DomainModule
from src.domains.dao.alignment import SemanticAlignmentJudge
from src.domains.dao.risk_tiers import DAO_TIER_MAP, dao_risk_tier
from src.domains.dao.simulator import AnvilBackend, SimulationBackend, SimulationRunner
from src.domains.dao.validator import FunctionCallValidator

VERSION = "0.1.0"


def build(
    classifier: Any,
    simulation_backend: SimulationBackend | None = None,
    rpc_url: str = "http://localhost:8545",
    scammer_addresses: set[str] | None = None,
) -> DomainModule:
    """Construct the DAO domain module.

    Parameters
    ----------
    classifier: a SafetyClassifier-shaped object used by the alignment judge.
    simulation_backend: optional override; defaults to AnvilBackend(rpc_url).
    rpc_url: anvil/foundry/tenderly endpoint for the default backend.
    scammer_addresses: optional set of known-bad addresses for the validator.
    """
    backend = simulation_backend or AnvilBackend(rpc_url=rpc_url)

    return DomainModule(
        name="dao",
        version=VERSION,
        description="Three-stage DAO function-call verification (validate → simulate → judge)",
        stages=[
            FunctionCallValidator(scammer_addresses=scammer_addresses),
            SimulationRunner(backend=backend),
            SemanticAlignmentJudge(classifier=classifier),
        ],
        risk_tier_fn=dao_risk_tier,
        tier_to_mode=DAO_TIER_MAP,
    )
