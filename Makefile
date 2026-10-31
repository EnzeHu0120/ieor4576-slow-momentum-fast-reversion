PYTHON ?= $(if $(wildcard .venv-replication/bin/python),.venv-replication/bin/python,python3)
CONFIG ?= configs/config.yaml
.PHONY: reproduce fresh resume data features models results report test

reproduce:
	$(PYTHON) scripts/run_all.py --config $(CONFIG)

resume:
	$(PYTHON) scripts/run_all.py --config $(CONFIG) --prepared-features

fresh:
	$(PYTHON) scripts/run_all.py --config $(CONFIG) --fresh-models

data:
	$(PYTHON) scripts/01_get_data.py

features:
	$(PYTHON) scripts/02_build_dataset.py --config $(CONFIG)

models:
	$(PYTHON) scripts/03_replicate.py --config $(CONFIG)

results:
	$(PYTHON) scripts/04_extend.py --config $(CONFIG)

report:
	$(PYTHON) scripts/05_report.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q
