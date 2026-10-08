# Contributing

Issues and PRs are welcome. Maintenance is best-effort.

```bash
make setup-dev
make lint test docs-check
```

- Core (`src/proving_ground`) must never import `examples/` (a test enforces it).
- Anything simulated must be tagged `synthetic: true`.
- A change to gates, serving, registry, promotion or rollback must keep `tests/core/test_flow.py` (promote then roll back) green.
- Report-viewer changes must keep the offline, hostile-input and CSP tests green. The viewer must never write to the audit log or registry.
- Don't add secrets, accounts or network calls to the default path.
