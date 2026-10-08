# Proving Ground: a gated model release template. Every target is a thin wrapper over `proving-ground`.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PG = $(PY) -m proving_ground.cli
FAMILY ?= gbm
SCENARIO ?= young_driver_surge
REASON ?= manual rollback

.PHONY: help setup setup-dev check data data-synthetic train gate promote-shadow shadow-eval promote-canary canary-eval promote \
        rollback audit-verify serve traffic inject-drift monitor viewer viewer-screenshots demo demo-fast test lint typecheck docs-check clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | sort

setup:           ## create .venv and install (macOS + LightGBM also needs: brew install libomp)
	python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -e ".[claims,tracking,drift,data]"

setup-dev:       ## add dev tools (pytest, ruff, mypy, playwright)
	.venv/bin/pip install -q -e ".[claims,tracking,drift,data,dev]"

check:           ## environment sanity check
	$(PY) scripts/check_env.py

data:            ## fetch freMTPL2freq once (OpenML, checksum-verified) and clean it; offline afterwards
	$(PY) -m examples.claims_frequency.data fetch && $(PY) -m examples.claims_frequency.data prepare

data-synthetic:  ## no-network stand-in dataset (labeled synthetic)
	$(PY) -m examples.claims_frequency.data synthetic

train:           ## train a candidate: make train FAMILY=gbm
	$(PG) train --family $(FAMILY) --tune

gate:            ## run gates G1..G8 on the last trained candidate of FAMILY
	$(PG) gate --family $(FAMILY)

promote-shadow:  ## register the candidate (only if its gate report passed) and start shadow
	$(PG) promote-shadow --family $(FAMILY)

shadow-eval:     ## evaluate shadow traffic
	$(PG) shadow-eval --family $(FAMILY)

promote-canary:  ## start the canary stage
	$(PG) promote-canary --family $(FAMILY)

canary-eval:     ## evaluate canary guardrails (aborts to 0% on breach)
	$(PG) canary-eval --family $(FAMILY)

promote:         ## flip champion (requires passing gates, shadow and canary)
	$(PG) promote --family $(FAMILY)

rollback:        ## restore previous_champion: make rollback REASON="..."
	$(PG) rollback --reason "$(REASON)"

audit-verify:    ## verify the hash-chained audit log
	$(PG) audit-verify

serve:           ## run the FastAPI server on :8000
	$(PG) serve

traffic:         ## send simulated traffic (in-process)
	$(PG) traffic --windows 2 --per-window 1000

inject-drift:    ## send synthetic drift: make inject-drift SCENARIO=young_driver_surge
	$(PG) inject-drift --scenario $(SCENARIO)

monitor:         ## compute drift reports and recommendations
	$(PG) monitor

viewer:          ## build reports/viewer/index.html (static, offline)
	$(PG) viewer

viewer-screenshots: ## capture README screenshots (needs the dev extra)
	$(PY) scripts/viewer_screenshots.py reports/viewer/index.html docs/img

demo:            ## the whole cycle, offline after the one-time data fetch
	$(PG) demo

demo-fast:       ## shortened run on the synthetic stand-in (CI smoke test)
	$(PG) demo --fast

test:            ## unit and integration tests
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests examples scripts

typecheck:
	$(PY) -m mypy src

docs-check:      ## every make target referenced in README/docs exists
	$(PY) scripts/docs_check.py

clean:           ## remove generated state (registry, reports, logs, audit log)
	rm -rf registry reports logs audit/decisions.jsonl
