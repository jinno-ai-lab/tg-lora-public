# Default settings for evaluation and training
MODEL_PATH ?= .cache/mlx_models/Qwen--Qwen3.5-9B
HF_MODEL_PATH ?= Qwen/Qwen2.5-9B-Instruct
ADAPTER_PATH ?= 
MAX_EXAMPLES ?= 
CONFIG ?= configs/9b_baseline.yaml
OUTPUT_DIR_MLX ?= reports/downstream_eval_mlx
OUTPUT_DIR_PYTORCH ?= reports/downstream_eval_pytorch

.PHONY: train eval-latest run-all eval-llm-jp-eval-mlx eval-downstream-mlx eval-llm-jp-eval-pytorch eval-downstream-pytorch smoke-test-mlx smoke-test-eval-mlx smoke-test-eval-pytorch help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'

# ── Training & Workflow ───────────────────────────────────────────────────────

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

run-all: ## Run training and then automatically evaluate the resulting adapter. Usage: make run-all [CONFIG=...]
	@$(MAKE) train CONFIG=$(CONFIG)
	@$(MAKE) eval-latest

# ── MLX Evaluation (Apple Silicon Only) ──────────────────────────────────────

eval-llm-jp-eval-mlx: ## Evaluate MLX model/adapter on JGLUE benchmarks (llm-jp-eval). Usage: make eval-llm-jp-eval-mlx [MODEL_PATH=...] [ADAPTER_PATH=...] [MAX_EXAMPLES=50] [OUTPUT_DIR=...]
	python scripts/eval_llm_jp_eval_mlx.py \
		--model-path "$(MODEL_PATH)" \
		$(if $(ADAPTER_PATH),--adapter-path "$(ADAPTER_PATH)") \
		$(if $(MAX_EXAMPLES),--max-examples $(MAX_EXAMPLES)) \
		--output-dir "$(or $(OUTPUT_DIR),reports/llm_jp_eval_mlx)"

eval-downstream-mlx: ## Evaluate MLX model/adapter on downstream tasks. Usage: make eval-downstream-mlx [MODEL_PATH=...] [ADAPTER_PATH=...] [MAX_EXAMPLES=50] [OUTPUT_DIR=...]
	python scripts/eval_downstream_mlx.py \
		--model-path "$(MODEL_PATH)" \
		$(if $(ADAPTER_PATH),--adapter-path "$(ADAPTER_PATH)") \
		$(if $(MAX_EXAMPLES),--max-examples $(MAX_EXAMPLES)) \
		--output-dir "$(or $(OUTPUT_DIR),$(OUTPUT_DIR_MLX))"

# ── PyTorch Evaluation (Cross-Platform) ──────────────────────────────────────

eval-llm-jp-eval-pytorch: ## Evaluate PyTorch model/adapter on JGLUE benchmarks (llm-jp-eval). Usage: make eval-llm-jp-eval-pytorch [HF_MODEL_PATH=...] [ADAPTER_PATH=...] [MAX_EXAMPLES=50] [OUTPUT_DIR=...] [DEVICE=...]
	python scripts/eval_llm_jp_eval.py \
		--model-path "$(HF_MODEL_PATH)" \
		$(if $(ADAPTER_PATH),--adapter-path "$(ADAPTER_PATH)") \
		$(if $(MAX_EXAMPLES),--max-examples $(MAX_EXAMPLES)) \
		$(if $(DEVICE),--device $(DEVICE)) \
		--output-dir "$(or $(OUTPUT_DIR),reports/llm_jp_eval_pytorch)"

eval-downstream-pytorch: ## Evaluate PyTorch model/adapter on downstream tasks. Usage: make eval-downstream-pytorch [HF_MODEL_PATH=...] [ADAPTER_PATH=...] [MAX_EXAMPLES=50] [OUTPUT_DIR=...] [DEVICE=...]
	python scripts/eval_downstream.py \
		--model-path "$(HF_MODEL_PATH)" \
		$(if $(ADAPTER_PATH),--adapter-path "$(ADAPTER_PATH)") \
		$(if $(MAX_EXAMPLES),--max-examples $(MAX_EXAMPLES)) \
		$(if $(DEVICE),--device $(DEVICE)) \
		--output-dir "$(or $(OUTPUT_DIR),$(OUTPUT_DIR_PYTORCH))"

# ── Smoke Tests (Verification) ───────────────────────────────────────────────

smoke-test-mlx: ## Run full training (2 steps) + MLX evaluation smoke test on Apple Silicon
	@echo "=== Preparing Smoke Test Directory ==="
	@mkdir -p data/smoke configs
	@head -n 2 data/downstream/jp_capability.jsonl > data/smoke/train.jsonl
	@head -n 2 data/downstream/jp_capability.jsonl > data/smoke/valid.jsonl
	@echo "model: mlx-community/Qwen2.5-0.5B-Instruct-4bit" > configs/smoke_config.yaml
	@echo "data: data/smoke" >> configs/smoke_config.yaml
	@echo "train: true" >> configs/smoke_config.yaml
	@echo "iters: 2" >> configs/smoke_config.yaml
	@echo "batch_size: 1" >> configs/smoke_config.yaml
	@echo "grad_accumulation_steps: 1" >> configs/smoke_config.yaml
	@echo "max_seq_length: 256" >> configs/smoke_config.yaml
	@echo "learning_rate: 1.0e-5" >> configs/smoke_config.yaml
	@echo "steps_per_report: 1" >> configs/smoke_config.yaml
	@echo "steps_per_eval: 2" >> configs/smoke_config.yaml
	@echo "save_every: 2" >> configs/smoke_config.yaml
	@echo "adapter_path: runs/mlx_smoke_test" >> configs/smoke_config.yaml
	@echo "=== Running Training Smoke Test (2 steps) ==="
	python -m mlx_lm.lora --config configs/smoke_config.yaml
	@echo "=== Running Evaluation Smoke Test ==="
	$(MAKE) eval-llm-jp-eval-mlx MODEL_PATH="mlx-community/Qwen2.5-0.5B-Instruct-4bit" ADAPTER_PATH="runs/mlx_smoke_test" MAX_EXAMPLES=1 OUTPUT_DIR="reports/smoke_test"
	$(MAKE) eval-downstream-mlx MODEL_PATH="mlx-community/Qwen2.5-0.5B-Instruct-4bit" ADAPTER_PATH="runs/mlx_smoke_test" MAX_EXAMPLES=1 OUTPUT_DIR="reports/smoke_test"
	@echo "=== Cleaning Up Smoke Test Temporary Files ==="
	@rm -rf data/smoke configs/smoke_config.yaml runs/mlx_smoke_test
	@echo "=== Smoke Test Passed Successfully! ==="

smoke-test-eval-mlx: ## Run MLX evaluation smoke test (no training)
	@echo "=== Running MLX Evaluation Smoke Test (No Training) ==="
	$(MAKE) eval-llm-jp-eval-mlx MODEL_PATH="mlx-community/Qwen2.5-0.5B-Instruct-4bit" MAX_EXAMPLES=1 OUTPUT_DIR="reports/smoke_test"
	$(MAKE) eval-downstream-mlx MODEL_PATH="mlx-community/Qwen2.5-0.5B-Instruct-4bit" MAX_EXAMPLES=1 OUTPUT_DIR="reports/smoke_test"
	@echo "=== Cleaning Up Smoke Test Temporary Files ==="
	@rm -rf reports/smoke_test
	@echo "=== MLX Evaluation Smoke Test Passed Successfully! ==="

smoke-test-eval-pytorch: ## Run PyTorch evaluation smoke test (no training) on any device
	@echo "=== Running PyTorch Evaluation Smoke Test (No Training) ==="
	$(MAKE) eval-llm-jp-eval-pytorch HF_MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" MAX_EXAMPLES=1 OUTPUT_DIR="reports/smoke_test_pytorch"
	$(MAKE) eval-downstream-pytorch HF_MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" MAX_EXAMPLES=1 OUTPUT_DIR="reports/smoke_test_pytorch"
	@echo "=== Cleaning Up Smoke Test Temporary Files ==="
	@rm -rf reports/smoke_test_pytorch
	@echo "=== PyTorch Evaluation Smoke Test Passed Successfully! ==="
