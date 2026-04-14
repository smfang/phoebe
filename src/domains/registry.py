"""
Runtime registry of installed domain modules.

Modules can be registered programmatically at bootstrap or discovered via
Python entry points (`sara.domains` group). Registration is idempotent
per name with a warning on replacement.
"""

from __future__ import annotations

import logging
from typing import Any

from src.domains.base import DomainModule

logger = logging.getLogger(__name__)


class UnknownDomainError(KeyError):
    """Raised when a request targets a domain that isn't registered."""


class DomainRegistry:
    """Lookup + lifecycle for installed domain modules."""

    def __init__(self) -> None:
        self._modules: dict[str, DomainModule] = {}

    def register(self, module: DomainModule) -> None:
        if module.name in self._modules:
            existing = self._modules[module.name]
            logger.warning(
                "Replacing domain module %s: %s -> %s",
                module.name, existing.version, module.version,
            )
        else:
            logger.info("Registered domain module: %s@%s", module.name, module.version)
        self._modules[module.name] = module

    def unregister(self, name: str) -> None:
        if name in self._modules:
            del self._modules[name]
            logger.info("Unregistered domain module: %s", name)

    def get(self, name: str) -> DomainModule:
        if name not in self._modules:
            raise UnknownDomainError(name)
        return self._modules[name]

    def has(self, name: str) -> bool:
        return name in self._modules

    def list(self) -> list[dict[str, Any]]:
        """Summary of registered modules — safe for public API."""
        return [
            {
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "stages": m.stage_names(),
                "tiers": [t.value for t in m.tier_to_mode],
            }
            for m in self._modules.values()
        ]

    def names(self) -> list[str]:
        return list(self._modules.keys())

    def discover_entry_points(self, config: Any, group: str = "sara.domains") -> int:
        """Discover and register modules published via Python entry points.

        Each entry point should be a callable `build(config) -> DomainModule`.
        Returns the number of modules registered.
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            return 0

        try:
            eps = entry_points(group=group)
        except TypeError:
            # Python < 3.10 fallback
            eps = entry_points().get(group, [])  # type: ignore[attr-defined]

        count = 0
        for ep in eps:
            try:
                builder = ep.load()
                module = builder(config)
                self.register(module)
                count += 1
            except Exception:
                logger.exception("Failed to load domain entry point: %s", ep.name)
        return count
