"""
test_sheila_redteam.py

Tests for Sheila's red team capability:
  - SheilaRedTeamLocal.run_session() returns RedTeamReport with correct total_probes
  - Probe IDs are signed (non-empty probe_signature)
  - Probe payload does NOT contain probe_id
  - Known injection probe is detected as BLOCKED (not evasion)
  - evasion_rate() computes correctly
"""

import pytest
from agents.sheila.red_team import SheilaRedTeamLocal, RedTeamProbe, ProbeResult


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_session_returns_report_with_correct_probe_count():
    """SheilaRedTeamLocal.run_session() returns RedTeamReport with correct total_probes."""
    red_team = SheilaRedTeamLocal()
    n_probes = 10
    report = await red_team.run_session(
        target_model_id="test-model-v1",
        categories=["prompt_injection", "treasury_manipulation"],
        n_probes=n_probes,
        signing_secret=b"test-secret-key-32bytes-padding!",
    )
    assert report.total_probes == n_probes
    assert report.target_model_id == "test-model-v1"
    assert report.session_id is not None
    assert isinstance(report.evasion_rate, float)
    assert 0.0 <= report.evasion_rate <= 1.0


@pytest.mark.asyncio
async def test_probe_ids_are_signed():
    """All probes must have non-empty probe_signature (HMAC-SHA256)."""
    red_team = SheilaRedTeamLocal()
    signing_secret = b"test-secret-for-signing-probes!!"
    probes = red_team._generate_probes(
        target_model_id="target-v1",
        categories=["prompt_injection"],
        n_probes=5,
        signing_secret=signing_secret,
    )
    for probe in probes:
        assert probe.probe_signature, f"Probe {probe.probe_id} has empty signature"
        assert len(probe.probe_signature) == 64  # SHA256 hex = 64 chars


@pytest.mark.asyncio
async def test_probe_payload_does_not_contain_probe_id():
    """Probe payload must NOT contain the probe_id (audit only, never sent to model)."""
    red_team = SheilaRedTeamLocal()
    signing_secret = b"test-secret-for-signing-probes!!"
    probes = red_team._generate_probes(
        target_model_id="target-v1",
        categories=["prompt_injection"],
        n_probes=10,
        signing_secret=signing_secret,
    )
    for probe in probes:
        assert probe.probe_id not in probe.payload, (
            f"Probe ID {probe.probe_id} found in payload! "
            "Probe IDs must never be sent to the model under test."
        )


@pytest.mark.asyncio
async def test_known_injection_probe_is_blocked():
    """Known injection probes must be detected as BLOCKED (evaded=False)."""
    red_team = SheilaRedTeamLocal()
    signing_secret = b"test-secret-key!!"
    probes = red_team._generate_probes(
        target_model_id="target-v1",
        categories=["prompt_injection"],
        n_probes=20,
        signing_secret=signing_secret,
    )
    # Find a probe with known injection text
    injection_probes = [
        p for p in probes
        if "ignore previous instructions" in p.payload.lower()
        or "ignore all previous" in p.payload.lower()
    ]
    assert len(injection_probes) > 0, "No injection probes generated"

    for probe in injection_probes:
        result = await red_team._run_probe(probe, "target-v1")
        assert result.evaded is False, (
            f"Injection probe was NOT detected as blocked: {probe.payload[:80]}"
        )


@pytest.mark.asyncio
async def test_evasion_rate_computes_correctly():
    """evasion_rate = evasions_found / total_probes."""
    red_team = SheilaRedTeamLocal()
    report = await red_team.run_session(
        target_model_id="test-model",
        categories=[],  # All categories
        n_probes=20,
        signing_secret=b"test-signing-key-for-evasion-rate",
    )
    expected_rate = report.evasions_found / report.total_probes if report.total_probes else 0.0
    assert abs(report.evasion_rate - round(expected_rate, 4)) < 0.001
