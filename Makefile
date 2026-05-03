PYTHON := $(shell command -v python3 2>/dev/null || command -v python)

.PHONY: up down logs test validate print-models

## Print all configured model backends
print-models:
	@$(PYTHON) scripts/print_models.py

## Bring up the API gateway (prints models first)
up: print-models
	docker compose up -d

## Stop the API gateway
down:
	docker compose down

## Stream gateway logs
logs:
	docker compose logs -f

## Run the full test suite
test:
	$(PYTHON) -m pytest tests/ -v

## Validate all YAML profiles (no Docker launched)
validate:
	bash scripts/validate_profile.sh profiles/*.yaml

## Dry-run a specific profile (usage: make dry-run PROFILE=profiles/example-vllm-gpu.yaml)
dry-run:
	bash scripts/run_profile.sh --dry-run $(PROFILE)
