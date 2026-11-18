[build]
PYTHON ?= python3

.PHONY: help install test lint prepare prepare-smoke train train-smoke eval push clean

help:
	@echo "install        Abhängigkeiten + Paket (editable)"
	@echo "test           Unit-Tests"
	@echo "lint           ruff"
	@echo "prepare        voller Datensatz -> data/lfm3-ha-de"
	@echo "prepare-smoke  kleiner Datensatz für einen schnellen Lauf"
	@echo "train          LoRA-Training (2 Epochen)"
	@echo "train-smoke    1 Epoche auf 200 Zeilen"
	@echo "eval           Auswertung gegen das Testset"
	@echo "push           Adapter verschmelzen und zu HF hochladen"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

prepare:
	lfm3ha prepare

prepare-smoke:
	lfm3ha prepare --limit 300 --oasst-limit 100 --out-dir data/smoke

train:
	lfm3ha train --epochs 2

train-smoke:
	lfm3ha train --dataset data/smoke --epochs 1 --max-train-rows 200 --output-dir outputs/smoke

eval:
	lfm3ha eval --adapter outputs/lfm3-ha-de --limit 300 --out outputs/metrics.json

push:
	lfm3ha push --adapter outputs/lfm3-ha-de --repo-id marc04/lfm3-ha-de

clean:
	rm -rf outputs data/smoke .pytest_cache .ruff_cache
