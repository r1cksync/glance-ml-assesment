# fashion-retrieval — operational entry points.
#
# Windows users: make targets shell out to bash scripts; use the PowerShell
# twins directly instead where they exist:
#   scripts\teardown.ps1          (= make destroy)
#   scripts\deploy_frontend.ps1   (= make deploy-frontend)
# The python targets (test/index/eval/seed/push-artifacts) work unchanged in
# any shell.

PY ?= python

.PHONY: test index eval infra-up seed push-artifacts deploy-frontend destroy

test:
	$(PY) -m pytest -q

index:
	$(PY) scripts/run_indexing.py

eval:
	$(PY) -m eval.ablations && $(PY) -m eval.latency

infra-up:
	terraform -chdir=infra init && terraform -chdir=infra apply

seed:
	$(PY) scripts/seed_data.py --upload

push-artifacts:
	$(PY) scripts/sync_artifacts.py push

deploy-frontend:
	bash scripts/deploy_frontend.sh

destroy:
	bash scripts/teardown.sh
