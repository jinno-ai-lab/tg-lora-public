# Default settings for evaluation and training
MODEL_PATH ?= .cache/mlx_models/Qwen--Qwen3.5-9B
ADAPTER_PATH ?= 
MAX_EXAMPLES ?= 
CONFIG ?= configs/9b_baseline.yaml

.PHONY: train eval-latest run-all eval-llm-jp-eval-mlx eval-downstream-mlx help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

train: ## Run training. Usage: make train [CONFIG=configs/...]
	@if [ ! -f "scripts/train.py" ] && [ ! -f "scripts/train_mlx_lora.py" ]; then \
		echo "Error: No training script found under scripts/ (expected scripts/train.py or scripts/train_mlx_lora.py). Please copy your training script."; \
		exit 1; \
	fi
	@if [ -f "scripts/train_mlx_lora.py" ]; then \
		python scripts/train_mlx_lora.py --config $(CONFIG); \
	else \
		python scripts/train.py --config $(CONFIG); \
	fi

eval-latest: ## Detect the latest run in runs/ and run downstream evaluations
	@LATEST_RUN=$$(ls -td runs/mlx_* 2>/dev/null | head -n 1); \
	if [ -z "$$LATEST_RUN" ]; then \
		echo "Error: No runs found in runs/ directory."; \
		exit 1; \
	fi; \
	echo "Latest run detected: $$LATEST_RUN"; \
	$(MAKE) eval-llm-jp-eval-mlx ADAPTER_PATH="$$LATEST_RUN"; \
	$(MAKE) eval-downstream-mlx ADAPTER_PATH="$$LATEST_RUN"

run-all: ## Run training and then automatically evaluate the resulting adapter. Usage: make run-all [CONFIG=configs/...]
	@$(MAKE) train CONFIG=$(CONFIG)
	@$(MAKE) eval-latest

eval-llm-jp-eval-mlx: ## Evaluate model & optional adapter on JGLUE benchmarks (llm-jp-eval). Usage: make eval-llm-jp-eval-mlx [MODEL_PATH=...] [ADAPTER_PATH=...] [MAX_EXAMPLES=50]
	python scripts/eval_llm_jp_eval_mlx.py \
		--model-path "$(MODEL_PATH)" \
		$(if $(ADAPTER_PATH),--adapter-path "$(ADAPTER_PATH)") \
		$(if $(MAX_EXAMPLES),--max-examples $(MAX_EXAMPLES))

eval-downstream-mlx: ## Evaluate model & optional adapter on downstream tasks (Japanese Capability & JSON compliance). Usage: make eval-downstream-mlx [MODEL_PATH=...] [ADAPTER_PATH=...] [MAX_EXAMPLES=50]
	python scripts/eval_downstream_mlx.py \
		--model-path "$(MODEL_PATH)" \
		$(if $(ADAPTER_PATH),--adapter-path "$(ADAPTER_PATH)") \
		$(if $(MAX_EXAMPLES),--max-examples $(MAX_EXAMPLES))
