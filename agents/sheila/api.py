"""
Sheila public API — the ONLY interface Sara uses to call Sheila.

All methods here are designed to be network-transparent:
  - Local (now): direct Python call
  - Phase 5 (TEE): same interface, backed by HTTP A2A to enclave

Sara code imports ONLY from this module, never from sheila internals.
"""

from dataclasses import dataclass
from typing import Literal, List, Optional
import os


@dataclass
class SheilaVerdict:
    turn_id: str
    decision: Literal["violation", "clean", "borderline"]
    category: Optional[str]          # Sara's 12-category taxonomy
    confidence: float                 # 0.0–1.0
    evidence: str                     # human-readable reasoning
    atlas_tactic: Optional[str]       # MITRE ATLAS tactic if applicable
    attestation_id: Optional[str]     # set when L2 attestation is active


@dataclass
class RedTeamReport:
    session_id: str
    target_model_id: str
    total_probes: int
    evasions_found: int
    evasion_rate: float
    atlas_coverage: dict
    top_evasions: List[dict]


class SheilaJudge:
    """
    Sheila in judge mode.
    Evaluates an agent interaction and returns a verdict.

    Phase 5: backed by SHEILA_A2A_URL env var when set.
    """

    def __init__(self):
        self._a2a_url = os.getenv("SHEILA_A2A_URL")
        if self._a2a_url:
            self._backend = self._make_a2a_backend()
        else:
            from agents.sheila.judge import SheilaJudgeLocal
            self._backend = SheilaJudgeLocal()

    async def judge(
        self,
        turn_id: str,
        user_input: str,
        agent_response: str,
        tool_calls: list = None,
        thinking_trace: str = None,      # CoT if available
        categories: List[str] = None,    # override default category set
        mode: Literal["judge", "redteam", "admin"] = "judge",
    ) -> SheilaVerdict:
        return await self._backend.judge(
            turn_id=turn_id,
            user_input=user_input,
            agent_response=agent_response,
            tool_calls=tool_calls or [],
            thinking_trace=thinking_trace,
            categories=categories,
            mode=mode,
        )

    def _make_a2a_backend(self):
        from agents.sheila.a2a_client import SheilaA2AClient
        return SheilaA2AClient(self._a2a_url)


class SheilaRedTeam:
    """
    Sheila in redteam mode.
    Generates adversarial probes against a target and returns a report.

    Phase 5: backed by SHEILA_A2A_URL env var when set.
    """

    def __init__(self):
        self._a2a_url = os.getenv("SHEILA_A2A_URL")
        if self._a2a_url:
            from agents.sheila.a2a_client import SheilaA2ARedTeamClient
            self._backend = SheilaA2ARedTeamClient(self._a2a_url)
        else:
            from agents.sheila.red_team import SheilaRedTeamLocal
            self._backend = SheilaRedTeamLocal()

    async def run_session(
        self,
        target_model_id: str,
        categories: List[str],
        n_probes: int = 50,
        signing_secret: bytes = None,
    ) -> RedTeamReport:
        return await self._backend.run_session(
            target_model_id=target_model_id,
            categories=categories,
            n_probes=n_probes,
            signing_secret=signing_secret,
        )
