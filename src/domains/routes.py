"""Starlette routes for the domain enforcement API.

Mounted into the existing arena server. Three classes of routes:

  - GET  /api/domains                          list registered modules
  - GET  /api/domains/{name}/healthcheck       per-module health
  - POST /api/domains/{name}/evaluate          run the module on a payload
  - GET  /api/domains/evaluations              recent decisions across all domains
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from src.domains.base import EvalRequest
from src.domains.enforcement import DomainEnforcement
from src.domains.registry import DomainRegistry, UnknownDomainError

logger = logging.getLogger(__name__)


class DomainRoutes:
    """HTTP surface for the domain enforcement framework."""

    def __init__(
        self,
        registry: DomainRegistry,
        enforcement: DomainEnforcement,
        store: Any | None = None,
    ) -> None:
        self._registry = registry
        self._enforcement = enforcement
        self._store = store

    def routes(self) -> list[Route]:
        return [
            Route("/api/domains", self._list_domains, methods=["GET"]),
            Route("/api/domains/evaluations", self._list_evaluations, methods=["GET"]),
            Route(
                "/api/domains/{name}/healthcheck",
                self._healthcheck,
                methods=["GET"],
            ),
            Route(
                "/api/domains/{name}/evaluate",
                self._evaluate,
                methods=["POST"],
            ),
        ]

    # ------------------------------------------------------------------

    async def _list_domains(self, request: Request) -> Response:
        return JSONResponse({"domains": self._registry.list()})

    async def _healthcheck(self, request: Request) -> Response:
        name = request.path_params["name"]
        try:
            module = self._registry.get(name)
        except UnknownDomainError:
            return JSONResponse({"error": f"unknown domain: {name}"}, status_code=404)

        if module.healthcheck_fn is None:
            return JSONResponse({"name": name, "version": module.version, "status": "ok"})
        try:
            payload = await module.healthcheck_fn()
        except Exception as exc:
            return JSONResponse(
                {"name": name, "version": module.version, "status": "error", "error": str(exc)},
                status_code=503,
            )
        return JSONResponse({"name": name, "version": module.version, "status": "ok", **payload})

    async def _evaluate(self, request: Request) -> Response:
        name = request.path_params["name"]
        if not self._registry.has(name):
            return JSONResponse({"error": f"unknown domain: {name}"}, status_code=404)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        try:
            req = EvalRequest(
                domain=name,
                payload=body.get("payload", {}),
                user_intent=body.get("user_intent", ""),
                caller=body.get("caller", ""),
                content_type=body.get("content_type", "application/json"),
            )
        except Exception as exc:
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=400)

        decision = await self._enforcement.evaluate(req)
        return JSONResponse(decision.model_dump())

    async def _list_evaluations(self, request: Request) -> Response:
        if self._store is None:
            return JSONResponse({"evaluations": [], "count": 0})

        domain = request.query_params.get("domain")
        limit = min(int(request.query_params.get("limit", "100")), 1000)

        conditions = ["1=1"]
        if domain:
            conditions.append(f"domain = '{_esc(domain)}'")

        sql = f"""
            SELECT request_id, domain, module_version, tier, mode, action,
                   severity, reason, decided_at, elapsed_ms
            FROM arena.domain_evaluations
            WHERE {" AND ".join(conditions)}
            ORDER BY decided_at DESC
            LIMIT {limit}
        """

        try:
            resp = await self._store.query(sql)
            evaluations = []
            for row in resp.result_rows:  # type: ignore[attr-defined]
                evaluations.append({
                    "request_id": str(row[0]),
                    "domain": str(row[1]),
                    "module_version": str(row[2]),
                    "tier": str(row[3]),
                    "mode": str(row[4]),
                    "action": str(row[5]),
                    "severity": int(row[6]),
                    "reason": str(row[7]),
                    "decided_at": float(row[8]),
                    "elapsed_ms": float(row[9]),
                })
            return JSONResponse({"evaluations": evaluations, "count": len(evaluations)})
        except Exception as exc:
            logger.warning("Failed to query domain_evaluations: %s", exc)
            return JSONResponse({"evaluations": [], "count": 0})


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("'", "\\'")
