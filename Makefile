PYTHON ?= python3
ROOT := $(CURDIR)
RUNS ?= $(ROOT)/runs
MODELS ?= $(ROOT)/models/model-paths.json
QWEN_LOAD ?= 4bit

.PHONY: help install models verify prepare-v1 train-v1 prepare-v2 train-v2 infer-validation test-media infer-test external-data infer-external

help:
	@echo 'make install             Install the Python dependencies in your active environment.'
	@echo 'make models              Download the pinned Qwen and CLAP models.'
	@echo 'make verify              Check metadata, released weights, model behavior and saved results on CPU.'
	@echo 'make prepare-v1          Fetch development media and cache the ten-second features on CUDA.'
	@echo 'make train-v1            Train the ten-second fusion and matched Qwen arms on CUDA.'
	@echo 'make prepare-v2          Fetch development media and cache the one-second features on CUDA.'
	@echo 'make train-v2            Train the one-second fusion and matched Qwen arms on CUDA.'
	@echo 'make infer-validation    Run and score the released models on validation in bf16.'
	@echo 'make test-media          Fetch the 53 test recordings for the published one-second evaluation.'
	@echo 'make infer-test          Run and score every one-second test window and mask on CUDA.'
	@echo 'make external-data       Download and extract RFVS and Halmstad.'
	@echo 'make infer-external      Run and score the three external datasets on CUDA.'

install:
	"$(PYTHON)" -m pip install -r requirements.txt

models:
	"$(PYTHON)" scripts/setup/download_models.py

verify:
	"$(PYTHON)" tests/check_composite_dataset.py "$(ROOT)"
	"$(PYTHON)" tests/check_window_composite_adapter.py "$(ROOT)"
	"$(PYTHON)" tests/check_fusion_baseline.py
	"$(PYTHON)" tests/check_trident_test_v2.py
	"$(PYTHON)" tests/check_release_weights.py
	"$(PYTHON)" scripts/verify_results.py

prepare-v1:
	PYTHON="$(PYTHON)" RUNS="$(RUNS)" MODELS="$(MODELS)" bash scripts/prepare.sh v1

prepare-v2:
	PYTHON="$(PYTHON)" RUNS="$(RUNS)" MODELS="$(MODELS)" bash scripts/prepare.sh v2

train-v1:
	PYTHON="$(PYTHON)" RUNS="$(RUNS)" MODELS="$(MODELS)" bash scripts/train.sh v1

train-v2:
	PYTHON="$(PYTHON)" RUNS="$(RUNS)" MODELS="$(MODELS)" bash scripts/train.sh v2

infer-validation:
	mkdir -p "$(RUNS)/validation-v2"
	"$(PYTHON)" scripts/infer/run_pipeline.py --dataset trident --output "$(RUNS)/validation-v2/predictions.json" --qwen-load bf16 --cache "$(RUNS)/validation-v2/features"
	"$(PYTHON)" scripts/infer/evaluate_external.py --predictions "$(RUNS)/validation-v2/predictions.json" --output "$(RUNS)/validation-v2/metrics.json"

test-media:
	"$(PYTHON)" scripts/infer/prepare_trident_test_v2.py
	"$(PYTHON)" scripts/infer/download_trident_test_v2.py

infer-test:
	mkdir -p "$(RUNS)/test-v2"
	"$(PYTHON)" scripts/infer/run_pipeline.py --dataset trident-test --output "$(RUNS)/test-v2/predictions.json" --qwen-load 4bit --cache "$(RUNS)/test-v2/features"
	"$(PYTHON)" scripts/infer/evaluate_external.py --predictions "$(RUNS)/test-v2/predictions.json" --output "$(RUNS)/test-v2/metrics.json"

external-data:
	"$(PYTHON)" scripts/setup/download_external.py rfvs
	"$(PYTHON)" scripts/setup/download_external.py halmstad

infer-external:
	mkdir -p "$(RUNS)/external"
	@set -eu; for dataset in rfvs halmstad-video halmstad-audio; do \
		"$(PYTHON)" scripts/infer/run_pipeline.py --dataset $$dataset --output "$(RUNS)/external/$$dataset-predictions.json" --qwen-load "$(QWEN_LOAD)" --cache "$(RUNS)/external/$$dataset-features"; \
		"$(PYTHON)" scripts/infer/evaluate_external.py --predictions "$(RUNS)/external/$$dataset-predictions.json" --output "$(RUNS)/external/$$dataset-metrics.json"; \
	done
