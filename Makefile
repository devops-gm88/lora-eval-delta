PY          ?= .venv/bin/python
MLX_LORA    ?= .venv/bin/mlx_lm.lora
export HF_HOME     := $(CURDIR)/.cache/huggingface
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
SHELL := /bin/bash

.DEFAULT_GOAL := help

help:
	@echo "make setup   create .venv and install requirements"
	@echo "make data    regenerate the synthetic dataset (deterministic) and check the hashes"
	@echo "make train   LoRA fine-tune per config/lora_config.yaml; writes logs/train.log and the loss curve"
	@echo "make eval    base vs adapter on the 100 test items; writes results/before_after.json"
	@echo "make demo    base vs adapter on 10 held-out documents"
	@echo "make test    run the unit tests"
	@echo "make clean   remove adapters, logs and generated results"

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-dev.txt

data:
	$(PY) scripts/generate_data.py
	$(PY) scripts/generate_data.py --check

train:
	@mkdir -p logs
	@set -o pipefail; $(MLX_LORA) -c config/lora_config.yaml 2>&1 | tee logs/train.log
	$(PY) scripts/plot_loss.py logs/train.log

eval:
	$(PY) scripts/evaluate.py

demo:
	$(PY) scripts/demo.py

test:
	$(PY) -m pytest

clean:
	rm -rf adapters logs/train.log results/loss.csv results/loss_curve.png

.PHONY: help setup data train eval demo test clean
