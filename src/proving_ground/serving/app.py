"""FastAPI serving wrapper: validation, shadow/canary routing, request logging, hot reload."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..candidate import Candidate
from ..provenance import utc_now
from ..workspace import Workspace

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=5000)
    window: int = 0                   # simulation window (set by the traffic driver)
    synthetic: bool = False           # tagged by the simulator; always propagated to logs
    scenario: str | None = None


class FaultRequest(BaseModel):
    candidate_error_rate: float = Field(ge=0, le=1)


def bucket(key: str) -> int:
    """Stable 0..99 bucket for a request key (same key -> same route, across processes)."""
    return int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16) % 100


class ServingState:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.lock = threading.Lock()
        self.champion: Candidate | None = None
        self.candidate: Candidate | None = None
        self.stage = "none"
        self.canary_pct = 0
        self.fault_rate = 0.0
        self.counters = {"requests": 0, "records": 0, "invalid": 0}
        self.schema = ws.project.schema()
        self.reload()

    def reload(self) -> None:
        reg = self.ws.registry
        al, st = reg.aliases(), reg.state()
        with self.lock:
            self.champion = reg.load(al["champion"]) if al["champion"] else None
            self.candidate = reg.load(al["candidate"]) if al["candidate"] else None
            self.stage, self.canary_pct = st["stage"], int(st["canary_pct"])

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok", "project": self.ws.project.name,
            "champion_hash": self.champion.hash if self.champion else None,
            "candidate_hash": self.candidate.hash if self.candidate else None,
            "stage": self.stage, "canary_pct": self.canary_pct,
            "gate_config_hash": self.ws.gate_config_hash(), "monitoring_config_hash": self.ws.monitoring_config_hash(),
        }

    # ------------------------------------------------------------------ scoring
    def _validate(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[int, str]]:
        """Return a boolean validity mask and per-row error text. Never raises."""
        bad: dict[int, str] = {}
        try:
            self.schema.validate(df, lazy=True)
        except Exception as exc:  # SchemaErrors / SchemaError
            fc = getattr(exc, "failure_cases", None)
            if fc is None or fc["index"].isna().any():
                msg = f"schema violation: {str(exc)[:160]}"
                if fc is not None:
                    msg = "schema violation: " + "; ".join(sorted({f"{r.column}: {r.check}" for r in fc.itertuples()})[:4])
                return np.zeros(len(df), bool), {i: msg for i in range(len(df))}
            for r in fc.itertuples():
                bad[int(r.index)] = f"{r.column}: {r.check}"
        mask = np.ones(len(df), bool)
        for i in bad:
            mask[i] = False
        return mask, bad

    def predict(self, req: PredictRequest) -> list[dict[str, Any]]:
        if self.champion is None:
            raise HTTPException(503, "no champion model is loaded")
        recs = req.records
        keys = [str(r.get("key", uuid.uuid4().hex[:10])) for r in recs]
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "key"} for r in recs])
        feats = self.ws.bundle().features
        mask, bad = self._validate(df) if all(c in df.columns for c in feats) else (
            np.zeros(len(df), bool), {i: "missing required columns" for i in range(len(df))})
        stage, champ, cand = self.stage, self.champion, self.candidate
        out: list[dict[str, Any]] = [{} for _ in recs]
        logs: list[dict[str, Any]] = []
        valid_idx = np.flatnonzero(mask)
        cp = dp = None
        c_ms = d_ms = 0.0
        if len(valid_idx):
            X = df.iloc[valid_idx][feats]
            t0 = time.perf_counter()
            cp = np.asarray(self.ws.project.predict(champ.model, X), float)
            c_ms = (time.perf_counter() - t0) * 1000 / len(valid_idx)
            if cand is not None and stage in ("shadow", "canary"):
                t0 = time.perf_counter()
                try:
                    dp = np.asarray(self.ws.project.predict(cand.model, X), float)
                except Exception:
                    dp = np.full(len(valid_idx), np.nan)
                d_ms = (time.perf_counter() - t0) * 1000 / len(valid_idx)
        pos = {int(i): j for j, i in enumerate(valid_idx)}
        for i, key in enumerate(keys):
            rid = uuid.uuid4().hex[:12]
            base = {"ts": utc_now(), "request_id": rid, "key": key, "window": req.window, "synthetic": req.synthetic,
                    "scenario": req.scenario, "stage": stage, "champion_hash": champ.hash,
                    "candidate_hash": cand.hash if cand else None, "features": _jsonable(recs[i])}
            if i not in pos:
                self.counters["invalid"] += 1
                out[i] = {"key": key, "request_id": rid, "error": {"type": "validation", "detail": bad.get(i, "invalid")}}
                logs.append({**base, "invalid": True, "invalid_reason": bad.get(i, "invalid")})
                continue
            j = pos[i]
            route, pred, model_hash = "champion", float(cp[j]), champ.hash
            cand_pred, cand_err = None, False
            if dp is not None:
                injected = self.fault_rate > 0 and (int(hashlib.sha256((key + "fault").encode()).hexdigest()[:8], 16) % 1000) < self.fault_rate * 1000
                if injected or np.isnan(dp[j]):
                    cand_err = True
                else:
                    cand_pred = float(dp[j])
                if stage == "canary" and bucket(key) < self.canary_pct:
                    if cand_err:
                        route = "canary_error"   # candidate failed for this user; champion answer served as fallback
                    else:
                        route, pred, model_hash = "canary", cand_pred, cand.hash
            out[i] = {"key": key, "request_id": rid, "prediction": pred, "route": route, "model_hash": model_hash}
            logs.append({**base, "invalid": False, "route": route, "champion_pred": float(cp[j]), "candidate_pred": cand_pred,
                         "candidate_error": cand_err, "champion_ms": c_ms, "candidate_ms": d_ms if dp is not None else None})
        self.counters["requests"] += 1
        self.counters["records"] += len(recs)
        self._append(logs)
        return out

    def _append(self, rows: list[dict[str, Any]]) -> None:
        self.ws.logs_dir.mkdir(parents=True, exist_ok=True)
        with self.lock, (self.ws.logs_dir / "requests.jsonl").open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")


def _jsonable(rec: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in rec.items():
        if k == "key":
            continue
        out[k] = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
    return out


def create_app(ws: Workspace) -> FastAPI:
    app = FastAPI(title=f"Proving Ground serving: {ws.project.name}", version="0.1.0")
    state = ServingState(ws)
    app.state.srv = state
    token = lambda: os.environ.get("PG_ADMIN_TOKEN", "demo-token")  # noqa: E731

    def admin(request: Request, x_admin_token: str | None) -> None:
        host = request.client.host if request.client else ""
        if host not in LOCAL_HOSTS or x_admin_token != token():
            raise HTTPException(403, "admin endpoints are local-only and need X-Admin-Token")

    @app.get("/health")
    def health():
        return state.health()

    @app.get("/metrics")
    def metrics():
        return state.counters

    @app.post("/predict")
    def predict(req: PredictRequest):
        return {"predictions": state.predict(req)}

    @app.post("/admin/reload")
    def reload(request: Request, x_admin_token: str | None = Header(default=None)):
        admin(request, x_admin_token)
        state.reload()
        return state.health()

    @app.post("/admin/fault")
    def fault(body: FaultRequest, request: Request, x_admin_token: str | None = Header(default=None)):
        admin(request, x_admin_token)
        state.fault_rate = body.candidate_error_rate
        return {"candidate_error_rate": state.fault_rate}

    @app.exception_handler(HTTPException)
    async def http_exc(_: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": {"type": "http", "detail": exc.detail}})

    return app


def attach_test_client(ws: Workspace):
    """In-process client wired into the workspace's reload/health hooks (used by the demo and tests)."""
    from fastapi.testclient import TestClient

    client = TestClient(create_app(ws))
    hdr = {"X-Admin-Token": os.environ.get("PG_ADMIN_TOKEN", "demo-token")}
    ws.reload_fn = lambda: client.post("/admin/reload", headers=hdr).raise_for_status()
    ws.health_fn = lambda: client.get("/health").json()
    return client
