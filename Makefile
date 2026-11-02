PYTHON ?= python3
CONFIG ?= configs/author_five_assets.yaml
.PHONY: data features reproduce results report test
data:
	$(PYTHON) scripts/01_get_data.py
features:
	$(PYTHON) scripts/02_build_dataset.py
reproduce:
	$(PYTHON) scripts/03_replicate.py --config $(CONFIG)
results:
	$(PYTHON) scripts/04_extend.py --config $(CONFIG)
report:
	$(PYTHON) scripts/05_report.py
test:
	$(PYTHON) -m pytest -q
