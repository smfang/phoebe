"""Tests for the DomainRegistry."""

import pytest

from src.domains.base import DomainModule, EnforcementMode, RiskTier
from src.domains.registry import DomainRegistry, UnknownDomainError


def _stub_module(name: str = "test", version: str = "0.1.0") -> DomainModule:
    return DomainModule(
        name=name,
        version=version,
        description="stub",
        stages=[],
        risk_tier_fn=lambda req: RiskTier.T1,
        tier_to_mode={RiskTier.T1: EnforcementMode.ASYNC},
    )


def test_register_and_get():
    reg = DomainRegistry()
    mod = _stub_module()
    reg.register(mod)
    assert reg.get("test") is mod


def test_get_unknown_raises():
    reg = DomainRegistry()
    with pytest.raises(UnknownDomainError):
        reg.get("nonexistent")


def test_has_returns_correct_membership():
    reg = DomainRegistry()
    assert reg.has("test") is False
    reg.register(_stub_module())
    assert reg.has("test") is True


def test_re_register_replaces():
    reg = DomainRegistry()
    reg.register(_stub_module(version="0.1.0"))
    reg.register(_stub_module(version="0.2.0"))
    assert reg.get("test").version == "0.2.0"


def test_unregister_removes():
    reg = DomainRegistry()
    reg.register(_stub_module())
    reg.unregister("test")
    assert not reg.has("test")


def test_unregister_missing_is_noop():
    reg = DomainRegistry()
    reg.unregister("nope")  # should not raise
    assert reg.names() == []


def test_list_returns_summary():
    reg = DomainRegistry()
    reg.register(_stub_module(name="a"))
    reg.register(_stub_module(name="b"))
    summary = reg.list()
    names = [s["name"] for s in summary]
    assert sorted(names) == ["a", "b"]
    for entry in summary:
        assert "version" in entry
        assert "stages" in entry
        assert "tiers" in entry


def test_names_lists_registered():
    reg = DomainRegistry()
    reg.register(_stub_module(name="x"))
    assert reg.names() == ["x"]
