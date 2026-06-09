"""
test_two_agent_boundary.py

Verifies the Sara↔Sheila architectural boundary:
  - Sara's src/safety/monitor.py does NOT import from agents/sheila/judge.py
    or agents/sheila/red_team.py directly
  - All communication goes through agents/sheila/api.py
  - SheilaJudge instantiates correctly
  - SheilaVerdict has all required fields
"""

import ast
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent


# ── Tests: import boundary ────────────────────────────────────────────────────


def _get_imports(filepath: Path) -> list[str]:
    """Return a list of all imported module paths in a Python file."""
    tree = ast.parse(filepath.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_monitor_does_not_import_sheila_judge():
    """
    Sara's monitor.py must NOT import from agents.sheila.judge directly.
    All Sara↔Sheila calls go through agents.sheila.api.
    """
    monitor_path = PROJECT_ROOT / "src" / "safety" / "monitor.py"
    assert monitor_path.exists(), f"monitor.py not found at {monitor_path}"

    imports = _get_imports(monitor_path)
    for imp in imports:
        assert "agents.sheila.judge" not in imp, (
            f"monitor.py imports from agents.sheila.judge directly: {imp}\n"
            "All Sara↔Sheila communication must go through agents.sheila.api"
        )


def test_monitor_does_not_import_sheila_red_team():
    """
    Sara's monitor.py must NOT import from agents.sheila.red_team directly.
    """
    monitor_path = PROJECT_ROOT / "src" / "safety" / "monitor.py"
    imports = _get_imports(monitor_path)
    for imp in imports:
        assert "agents.sheila.red_team" not in imp, (
            f"monitor.py imports from agents.sheila.red_team directly: {imp}\n"
            "All Sara↔Sheila communication must go through agents.sheila.api"
        )


def test_sheila_api_module_exists():
    """agents/sheila/api.py must exist as the public interface."""
    api_path = PROJECT_ROOT / "agents" / "sheila" / "api.py"
    assert api_path.exists(), "agents/sheila/api.py not found"


# ── Tests: SheilaJudge instantiation ─────────────────────────────────────────


def test_sheila_judge_instantiates_local_without_a2a_url():
    """
    When SHEILA_A2A_URL is not set, SheilaJudge should instantiate SheilaJudgeLocal.
    """
    env = {k: v for k, v in os.environ.items() if k != "SHEILA_A2A_URL"}
    with patch.dict(os.environ, env, clear=True):
        from agents.sheila.api import SheilaJudge
        judge = SheilaJudge()
        # Backend should be SheilaJudgeLocal
        from agents.sheila.judge import SheilaJudgeLocal
        assert isinstance(judge._backend, SheilaJudgeLocal)


def test_sheila_judge_uses_a2a_when_url_set():
    """
    When SHEILA_A2A_URL is set, SheilaJudge should use the A2A backend.
    """
    with patch.dict(os.environ, {"SHEILA_A2A_URL": "http://enclave.test:8080"}):
        # Re-import to pick up new env var
        import importlib
        import agents.sheila.api as api_module
        importlib.reload(api_module)
        judge = api_module.SheilaJudge()
        from agents.sheila.a2a_client import SheilaA2AClient
        assert isinstance(judge._backend, SheilaA2AClient)
        # Restore
        importlib.reload(api_module)


# ── Tests: SheilaVerdict fields ───────────────────────────────────────────────


def test_sheila_verdict_has_required_fields():
    """SheilaVerdict dataclass must have all required fields."""
    from agents.sheila.api import SheilaVerdict
    import dataclasses

    fields = {f.name for f in dataclasses.fields(SheilaVerdict)}
    required = {
        "turn_id", "decision", "category", "confidence",
        "evidence", "atlas_tactic", "attestation_id",
    }
    assert required.issubset(fields), (
        f"SheilaVerdict missing fields: {required - fields}"
    )


def test_sheila_verdict_instantiates_correctly():
    """SheilaVerdict can be created with all fields."""
    from agents.sheila.api import SheilaVerdict
    verdict = SheilaVerdict(
        turn_id="test-turn-1",
        decision="clean",
        category=None,
        confidence=0.9,
        evidence="No violation detected",
        atlas_tactic=None,
        attestation_id=None,
    )
    assert verdict.turn_id == "test-turn-1"
    assert verdict.decision == "clean"
    assert verdict.confidence == 0.9
