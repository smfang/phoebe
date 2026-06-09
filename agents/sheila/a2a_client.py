"""
Sheila A2A (Agent-to-Agent) HTTP client stubs.

These classes implement the same interface as SheilaJudgeLocal and
SheilaRedTeamLocal but dispatch over HTTP to a remote Sheila enclave.

Phase 5: when SHEILA_A2A_URL env var is set, SheilaJudge and SheilaRedTeam
will use these backends instead of the local implementations.

Not implemented yet — raises NotImplementedError until Phase 5 TEE work.
"""

from typing import List, Literal, Optional


class SheilaA2AClient:
    """
    HTTP A2A backend for SheilaJudge.

    In Phase 5, Sheila runs inside a TEE enclave as a standalone FastAPI
    service. This client dispatches judge() calls over HTTP to that service.
    The interface is identical to SheilaJudgeLocal — zero changes to call sites.
    """

    def __init__(self, a2a_url: str):
        self._url = a2a_url

    async def judge(
        self,
        turn_id: str,
        user_input: str,
        agent_response: str,
        tool_calls: list = None,
        thinking_trace: str = None,
        categories: List[str] = None,
        mode: Literal["judge", "redteam", "admin"] = "judge",
    ):
        raise NotImplementedError(
            "SheilaA2AClient: Phase 5 TEE A2A not yet implemented. "
            f"Would POST to {self._url}/judge with turn_id={turn_id}"
        )


class SheilaA2ARedTeamClient:
    """
    HTTP A2A backend for SheilaRedTeam.

    In Phase 5, dispatches run_session() calls to the Sheila TEE enclave.
    The interface is identical to SheilaRedTeamLocal — zero changes to call sites.
    """

    def __init__(self, a2a_url: str):
        self._url = a2a_url

    async def run_session(
        self,
        target_model_id: str,
        categories: List[str],
        n_probes: int = 50,
        signing_secret: bytes = None,
    ):
        raise NotImplementedError(
            "SheilaA2ARedTeamClient: Phase 5 TEE A2A not yet implemented. "
            f"Would POST to {self._url}/redteam/session with target={target_model_id}"
        )
