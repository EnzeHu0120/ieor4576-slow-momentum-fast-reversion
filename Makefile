PYTHON ?= python3
.PHONY: demo test

demo:
	$(PYTHON) scripts/02_build_dataset.py --demo
	$(PYTHON) scripts/04_backtest.py --demo

test:
	$(PYTHON) -m pytest -q
