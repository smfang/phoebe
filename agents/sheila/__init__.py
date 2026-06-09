"""
Sheila — Red Team Judge Agent
Sara Platform v2.0

Sheila operates in three modes:
  judge    — evaluates Arena researcher submissions against target models
  redteam  — autonomously generates adversarial attacks targeting Sara's rule gaps
  admin    — manages bounties, reviews submissions, controls payouts

Sara communicates with Sheila exclusively through agents/sheila/api.py.
Sheila internals are never imported directly by Sara code.

Phase 5 note: this package will be extracted to a standalone FastAPI service
running inside a TEE enclave. The api.py interface is designed to be
network-transparent — swapping direct calls for A2A HTTP requires no
changes to Sara's call sites.
"""

from agents.sheila.api import SheilaJudge, SheilaRedTeam
__all__ = ["SheilaJudge", "SheilaRedTeam"]
