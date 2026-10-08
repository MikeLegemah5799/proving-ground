# Security

## Reporting
Please report vulnerabilities privately through GitHub's "Report a vulnerability" (private security advisory) on this repository.
Do not open a public issue for security problems.

## Known properties and limits
- **The registry pickles models.** Loading a pickle executes code. Only load a `registry/` directory you created. Never point the template at an untrusted registry.
- **The audit log is tamper-evident, not tamper-proof.** Anyone who can rewrite the whole file can recompute the hash chain. Anchor the head hash (CI artifact, signed commit) if you need stronger guarantees.
- **Admin endpoints** (`/admin/reload`, `/admin/fault`) are local-only and require `X-Admin-Token` (env `PG_ADMIN_TOKEN`, default `demo-token`). **Change the default before exposing the server anywhere.**
- **The report viewer treats its inputs as untrusted:** all data is escaped at build time, embedded JSON is serialized safely, a CSP forbids everything except the one inline script by hash, and there are no network requests.
- No secrets are needed or stored by default.
