# Runbook

## When to roll back
- The monitor reached `alert` and wrote a **hold** audit entry with a rollback recommendation, **and** a human agrees the previous champion is the safer state.
- A canary guardrail breach already aborts automatically (`canary_abort`, traffic back to 0%); no rollback needed, the champion never changed.
- Do **not** expect rollback to fix drifted inputs. It restores the previous known-good model; the data problem remains and needs its own fix.
- `auto_rollback: true` in `config/monitoring.yaml` makes the monitor roll back by itself. It is off by default.

## How
```bash
make audit-verify                       # is the evidence trustworthy?
make rollback REASON="drift alert 2026-.., see reports/drift/..."
make audit-verify
make viewer && open reports/viewer/index.html
```
`rollback` flips `champion ← previous_champion`, clears `previous_champion`, hot-reloads the server, verifies the served hash through `GET /health`,
and writes a `rollback` audit entry (with the health-check evidence). It is **idempotent**: a second call returns `already_rolled_back` and writes nothing.
If the server reports a different hash than the restored one, the audit entry is written with `[HEALTH VERIFICATION FAILED]` and the command exits non-zero: treat it as an incident.

## When the audit chain fails
`make audit-verify` names the first bad sequence number. Do not append (the log refuses). Restore the file from your anchor (CI artifact / signed commit) and investigate who had write access.

## Promoting a new model
```bash
make train FAMILY=gbm && make gate FAMILY=gbm
make promote-shadow FAMILY=gbm          # refuses unless the gate report passed
make traffic && make shadow-eval FAMILY=gbm
make promote-canary FAMILY=gbm
make traffic && make canary-eval FAMILY=gbm
make promote FAMILY=gbm
```
Against a running server, pass `--url http://127.0.0.1:8000` to `traffic`, `inject-drift` and `rollback`.
`make demo` does all of this, in-process, in about a minute and a half.

## Local server
`make serve`, then `curl localhost:8000/health`. Set `PG_ADMIN_TOKEN` before doing anything beyond localhost.
