.DEFAULT_GOAL := help
PY := .venv/bin/python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

setup: ## Create the venv (Python 3.12 via uv) and install dependencies
	@test -x "$(UV)" || curl -LsSf https://astral.sh/uv/install.sh | sh
	"$(UV)" venv --python 3.12 .venv
	"$(UV)" pip install -e . pytest
	@echo "OK. Now: cp .env.example .env and add your free GROQ + GEMINI keys."

data: ## Download the Kaggle dataset into data/raw/ (needs Kaggle creds)
	bash scripts/download_data.sh

scan-brands: ## Rank brands by resolvable-thread volume (helps you pick one)
	$(PY) -m trustagent.cli scan-brands

smoke: ## Fast end-to-end run on the tiny committed synthetic sample (no Kaggle, no keys)
	TRUSTAGENT_OFFLINE=1 $(PY) -m trustagent.cli build --smoke
	TRUSTAGENT_OFFLINE=1 $(PY) -m trustagent.cli eval --smoke
	@echo "smoke OK — see results/RESULTS.md (numbers are meaningless, this only proves the wiring)"

build: ## Ingest -> threads -> intents -> retrieval index (real data)
	$(PY) -m trustagent.cli build

golden: ## Assisted labelling workflow to build golden/golden_set.csv yourself
	$(PY) scripts/build_golden_set.py
	@echo "Next: $(PY) scripts/build_golden_set.py --review   (label every row)"

provisional-golden: ## Dev-only: auto-fill a PROVISIONAL golden set so `make all` runs before you label
	$(PY) scripts/build_golden_set.py --auto

eval: ## Run baselines + agent + LLM-judge, write results/
	$(PY) -m trustagent.cli eval

judge-validation: ## Compute how well the LLM judge agrees with your human ratings
	$(PY) scripts/judge_validation.py

failures: ## Regenerate results/failure_analysis.md
	$(PY) scripts/failure_analysis.py

report: ## Fill REPORT.md placeholders from results/metrics.json -> REPORT_GENERATED.md
	$(PY) scripts/render_report.py

demo: ## Interactive: type a customer message, see intent + evidence + draft + decision
	$(PY) -m trustagent.cli demo

all: build eval failures report ## Reproduce the headline results (target: <15 min)

test: ## Unit tests
	$(PY) -m pytest -q tests/

clean:
	rm -rf artifacts/* results/*.json results/*.png results/*.md .cache

.PHONY: help setup data scan-brands smoke build golden provisional-golden eval judge-validation failures report demo all test clean
