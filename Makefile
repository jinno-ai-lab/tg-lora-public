.PHONY: setup install inspect download-data prepare-data \
       smoke smoke-tg smoke-bl \
       train-baseline train-tg-lora train-tg-lora-optreuse train-tg-lora-prefix \
       eval eval-lora eval-downstream eval-downstream-mlx eval-llm-jp-eval-mlx eval-mlx eval-base eval-35b-base eval-35b-ckpt \
       ingest-evidence check-evidence run-paper-experiment freeze-validloss-ci freeze-validloss-ci-heterogeneous freeze-validloss-ci-generalize \

	compare compare-prefix compare-prefix-cold compare-prefix-warm compare-prefix-coldwarm compare-report paper-memory paper-memory-dry-run paper-memory-one-shot paper-memory-compare-modes paper-memory-all-modes paper-memory-evaluate-gates paper-memory-external-eval paper-memory-frontier-sweep paper-memory-cache-ablation cosine-n-ablation cosine-n-ablation-dry-run cosine-n-skip-ablation cosine-n-skip-ablation-dry-run precompute-prefix-cache ablation sweep accel-sweep \
	bench-optimizer bench-prefix-cache bench-prefix-cache-one-shot analyze-prefix-break-even bench-velocity-ops bench-velocity-ops-ci bench-velocity-ops-save-baseline \
       test test-accel test-cov test-integration test-trajectory test-cli-help lint format clean clean-data clean-runs \
       diagnose recover ci check-status \
       convert-mlx train-mlx train-mlx-baseline train-mlx-continuous train-mlx-upstream train-mlx-smoke mlx-data compare-mlx \
       help

PYTHON ?= python
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
NOW_STAMP := $(shell date +%Y%m%d_%H%M%S)

# Default config
CONFIG ?= configs/9b_tg_lora.yaml
BASE_MODEL ?= Qwen/Qwen3.5-9B
LM_EVAL_HARNESS ?= $(HOME)/lm-evaluation-harness
LM_EVAL_MODEL ?= .cache/mlx_models/$(shell echo $(BASE_MODEL) | sed 's/\//--/g')

# 35B MoE model for Track B
MODEL_35B ?= .cache/mlx_models/Qwen3.6-35B-A3B-4bit

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────────────────

setup: ## Full environment setup (conda + deps + latest transformers)
	bash scripts/setup_env.sh

install: ## Install into existing venv
	$(PYTHON) -m venv $(VENV) || true
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -e ".[dev]"
	$(PIP) install -e $(LM_EVAL_HARNESS)
	@echo "Done. Activate: source $(VENV)/bin/activate"

# ── Inspection ───────────────────────────────────────────────────────────────

inspect: ## Inspect model architecture & LoRA target module names
	$(PYTHON_VENV) scripts/inspect_model.py --model $(BASE_MODEL)

inspect-config: ## Inspect model from YAML config
	$(PYTHON_VENV) scripts/inspect_model.py --config $(CONFIG)

# ── Data ─────────────────────────────────────────────────────────────────────

download-data: ## Download public datasets (Dolly 15k, Capybara)
	$(PYTHON_VENV) scripts/download_data.py --dataset all --output-dir data/raw

download-dolly: ## Download Dolly 15k only
	$(PYTHON_VENV) scripts/download_data.py --dataset dolly --output-dir data/raw

download-capybara: ## Download Capybara only
	$(PYTHON_VENV) scripts/download_data.py --dataset capybara --output-dir data/raw

prepare-data: ## Prepare training data (Dolly 15k → train/valid split)
	$(PYTHON_VENV) scripts/prepare_data.py --source dolly --train-size 5000 --valid-size 500 --test-size 500

prepare-data-small: ## Prepare small dataset for quick testing (1k train)
	$(PYTHON_VENV) scripts/prepare_data.py --source dolly --train-size 1000 --valid-size 100

prepare-capybara: ## Prepare Capybara dataset
	$(PYTHON_VENV) scripts/prepare_data.py --source capybara --train-size 10000 --valid-size 1000

# ── Smoke tests (1 cycle, ~2-3 min each) ────────────────────────────────────

smoke: ## Run both TG-LoRA + baseline smoke tests (~5 min total)
	@echo "=== TG-LoRA smoke ===" && $(MAKE) smoke-tg
	@echo "=== Baseline smoke ===" && $(MAKE) smoke-bl

smoke-tg: ## 1-cycle TG-LoRA smoke test (~2 min)
	@mkdir -p runs/smoke_tg
	@cp configs/9b_tg_lora.yaml runs/smoke_tg/config.yaml
	$(PYTHON_VENV) -c "\
from omegaconf import OmegaConf;\
cfg = OmegaConf.load('runs/smoke_tg/config.yaml');\
cfg.training.max_cycles = 1;\
cfg.eval.full_eval_every_cycles = 9999;\
cfg.logging.save_every_cycles = 9999;\
cfg.logging.run_dir = 'runs/smoke_tg';\
cfg.data.max_seq_len = 1024;\
OmegaConf.save(cfg, 'runs/smoke_tg/config.yaml')"
	$(PYTHON_VENV) -m src.training.train_tg_lora --config runs/smoke_tg/config.yaml

smoke-bl: ## 8-step baseline smoke test (~2 min)
	@mkdir -p runs/smoke_bl
	@cp configs/9b_baseline.yaml runs/smoke_bl/config.yaml
	$(PYTHON_VENV) -c "\
from omegaconf import OmegaConf;\
cfg = OmegaConf.load('runs/smoke_bl/config.yaml');\
cfg.training.max_steps = 8;\
cfg.training.save_every_steps = 9999;\
cfg.eval.full_eval_every_steps = 9999;\
cfg.logging.run_dir = 'runs/smoke_bl';\
cfg.data.max_seq_len = 1024;\
OmegaConf.save(cfg, 'runs/smoke_bl/config.yaml')"
	$(PYTHON_VENV) -m src.training.train_baseline_qlora --config runs/smoke_bl/config.yaml

# ── Training (full configs) ─────────────────────────────────────────────────

train-baseline: ## Run QLoRA baseline training
	$(PYTHON_VENV) -m src.training.train_baseline_qlora --config configs/9b_baseline.yaml

train-tg-lora: ## Run TG-LoRA training (paper-PoC deterministic)
	$(PYTHON_VENV) -m src.training.train_tg_lora --config configs/9b_tg_lora.yaml

train-tg-lora-optreuse: ## TG-LoRA with experimental optimizer state reuse
	$(PYTHON_VENV) -m src.training.train_tg_lora --config configs/9b_tg_lora_optimizer_reuse_experimental.yaml

train-tg-lora-prefix: ## TG-LoRA with experimental prefix feature cache
	$(PYTHON_VENV) -m src.training.train_tg_lora --config configs/9b_tg_lora_prefix_feature_cache_experimental.yaml

# ── Evaluation ───────────────────────────────────────────────────────────────

EVAL_TASKS ?= arc_easy,hellaswag,truthfulqa_mc2
EVAL_OUTPUT ?= reports/eval
EVAL_LOG ?= reports/eval/logs

eval: ## Run lm-eval with MLX backend. ADAPTER_PATH= optional. MODEL= to override.
	@mkdir -p $(EVAL_OUTPUT) $(EVAL_LOG); \
	ARGS="model=$(LM_EVAL_MODEL)"; \
	CKPT="$(EVAL_OUTPUT)/ckpt_eval_$(NOW_STAMP).json"; \
	LOG="$(EVAL_LOG)/eval_$(NOW_STAMP).log"; \
	if [ -n "$(ADAPTER_PATH)" ]; then ARGS="$$ARGS,adapter_path=$(ADAPTER_PATH)"; \
		CKPT="$(EVAL_OUTPUT)/ckpt_$$(basename $(ADAPTER_PATH))_eval.json"; \
		LOG="$(EVAL_LOG)/$$(basename $(ADAPTER_PATH))_eval.log"; fi; \
	echo "[eval] output: $(EVAL_OUTPUT)/eval_$(NOW_STAMP).json"; \
	echo "[eval] log: $$LOG"; \
	echo "[eval] checkpoint: $$CKPT (resume-safe)"; \
	PYTHONPATH=$(LM_EVAL_HARNESS):$(PWD) \
	MLX_EVAL_CKPT_PATH="$$CKPT" \
	$(PYTHON_VENV) -m lm_eval \
		--model mlx \
		--model_args "$$ARGS" \
		--tasks $(EVAL_TASKS) \
		--batch_size 1 \
		--log_samples \
		--output_path $(EVAL_OUTPUT)/eval_$(NOW_STAMP).json \
		2>&1 | tee "$$LOG"; \
	rm -f "$$CKPT"

eval-mlx: ## Eval MLX model+adapter. ADAPTER_PATH= required. MODEL= to override.
	@if [ -z "$(ADAPTER_PATH)" ]; then \
		echo "Usage: make eval-mlx ADAPTER_PATH=runs/mlx_qlora_baseline_500"; \
		exit 1; \
	fi
	@mkdir -p $(EVAL_OUTPUT) $(EVAL_LOG); \
	CKPT="$(EVAL_OUTPUT)/ckpt_$$(basename $(ADAPTER_PATH)).json"; \
	LOG="$(EVAL_LOG)/$$(basename $(ADAPTER_PATH))_eval.log"; \
	OUT="$(EVAL_OUTPUT)/$$(basename $(ADAPTER_PATH))_eval.json"; \
	echo "[eval-mlx] model: $(LM_EVAL_MODEL)"; \
	echo "[eval-mlx] adapter: $(ADAPTER_PATH)"; \
	echo "[eval-mlx] output: $$OUT"; \
	echo "[eval-mlx] log: $$LOG (tail -f for progress)"; \
	echo "[eval-mlx] checkpoint: $$CKPT (resume-safe)"; \
	PYTHONPATH=$(LM_EVAL_HARNESS):$(PWD) \
	MLX_EVAL_CKPT_PATH="$$CKPT" \
	$(PYTHON_VENV) -m lm_eval \
		--model mlx \
		--model_args "model=$(LM_EVAL_MODEL),adapter_path=$(ADAPTER_PATH)" \
		--tasks $(EVAL_TASKS) \
		--batch_size 1 \
		--log_samples \
		--output_path "$$OUT" \
		2>&1 | tee "$$LOG"; \
	rm -f "$$CKPT"

eval-base: ## Eval base model only (no adapter). MODEL= to override.
	@mkdir -p $(EVAL_OUTPUT) $(EVAL_LOG); \
	LOG="$(EVAL_LOG)/base_model_eval.log"; \
	CKPT="$(EVAL_OUTPUT)/ckpt_base_model.json"; \
	echo "[eval-base] model: $(LM_EVAL_MODEL)"; \
	echo "[eval-base] log: $$LOG"; \
	PYTHONPATH=$(LM_EVAL_HARNESS):$(PWD) \
	MLX_EVAL_CKPT_PATH="$$CKPT" \
	$(PYTHON_VENV) -m lm_eval \
		--model mlx \
		--model_args "model=$(LM_EVAL_MODEL)" \
		--tasks $(EVAL_TASKS) \
		--batch_size 1 \
		--log_samples \
		--output_path $(EVAL_OUTPUT)/base_model_eval.json \
		2>&1 | tee "$$LOG"; \
	rm -f "$$CKPT"

eval-35b-base: ## Eval 35B base model (Track B)
	@$(MAKE) eval-base LM_EVAL_MODEL=$(MODEL_35B)

eval-35b-ckpt: ## Eval 35B adapter checkpoint. CKPT= required (e.g. CKPT=100)
	@if [ -z "$(CKPT)" ]; then \
		echo "Usage: make eval-35b-ckpt CKPT=100"; \
		exit 1; \
	fi
	@ADAPTER="runs/mlx_qlora_35b_baseline/ckpt_$(CKPT)"; \
	if [ ! -d "$$ADAPTER" ]; then echo "Error: $$ADAPTER not found"; exit 1; fi; \
	$(MAKE) eval-mlx LM_EVAL_MODEL=$(MODEL_35B) ADAPTER_PATH="$$ADAPTER"

eval-lora: ## Eval LoRA adapter via HF merge+eval (CUDA only, legacy)
	@if [ -z "$(ADAPTER_PATH)" ]; then \
		echo "Usage: make eval-lora ADAPTER_PATH=runs/<experiment>/checkpoint-<step>"; \
		exit 1; \
	fi
	bash scripts/run_eval_lora.sh $(BASE_MODEL) $(ADAPTER_PATH)

eval-downstream: ## Evaluate model and optional adapter on Japanese & JSON formatting downstream tasks. Usage: make eval-downstream [ADAPTER_PATH=runs/.../best_model] [DEVICE=cpu/mps/cuda] [MAX_EXAMPLES=5]
	$(PYTHON_VENV) scripts/eval_downstream.py \
		$(if $(ADAPTER_PATH),--adapter-path $(ADAPTER_PATH)) \
		$(if $(DEVICE),--device $(DEVICE)) \
		$(if $(MAX_EXAMPLES),--max-examples $(MAX_EXAMPLES)) \
		--config $(or $(CONFIG),configs/9b_tg_lora.yaml)

eval-downstream-mlx: ## Evaluate MLX model and optional adapter on Japanese & JSON formatting downstream tasks. Usage: make eval-downstream-mlx [ADAPTER_PATH=runs/mlx_...] [MODEL_PATH=.cache/mlx_models/...] [MAX_EXAMPLES=5]
	$(PYTHON_VENV) mlx/scripts/eval_downstream.py \
		$(if $(ADAPTER_PATH),--adapter-path $(ADAPTER_PATH)) \
		$(if $(MODEL_PATH),--model-path $(MODEL_PATH)) \
		$(if $(MAX_EXAMPLES),--max-examples $(MAX_EXAMPLES))

eval-llm-jp-eval-mlx: ## Evaluate MLX model and optional adapter on JGLUE benchmarks (llm-jp-eval). Usage: make eval-llm-jp-eval-mlx [ADAPTER_PATH=runs/mlx_...] [MODEL_PATH=.cache/mlx_models/...] [MAX_EXAMPLES=20]
	$(PYTHON_VENV) mlx/scripts/eval_llm_jp_eval.py \
		$(if $(ADAPTER_PATH),--adapter-path $(ADAPTER_PATH)) \
		$(if $(MODEL_PATH),--model-path $(MODEL_PATH)) \
		$(if $(MAX_EXAMPLES),--max-examples $(MAX_EXAMPLES))




# ── Comparison experiments ───────────────────────────────────────────────────

compare: ## Run fair comparison: baseline vs TG-LoRA (BUDGET=backward passes)
	CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),) \
	BUDGET_PASSES=$(or $(BUDGET),1500) \
	BASELINE_CONFIG=$(or $(BASELINE_CONFIG),configs/9b_baseline.yaml) \
	TG_LORA_CONFIG=$(or $(TG_LORA_CONFIG),configs/9b_tg_lora.yaml) \
	MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
	QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),64) \
	EVAL_POINTS=$(or $(EVAL_POINTS),4) \
	MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false) \
	TG_PREFIX_CACHE_DIR=$(or $(TG_PREFIX_CACHE_DIR),) \
	TG_PREFIX_FORCE_REBUILD=$(or $(TG_PREFIX_FORCE_REBUILD),) \
	bash scripts/run_comparison.sh

compare-prefix: ## Fair suffix-only comparison: cache-free baseline vs prefix-cache TG-LoRA
	CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),) \
	BUDGET_PASSES=$(or $(BUDGET),240) \
	BASELINE_CONFIG=configs/9b_baseline_suffix_only_last25.yaml \
	TG_LORA_CONFIG=configs/9b_tg_lora_prefix_feature_cache_experimental.yaml \
	MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
	QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
	EVAL_POINTS=$(or $(EVAL_POINTS),3) \
	MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false) \
	TG_PREFIX_CACHE_DIR=$(or $(CACHE_DIR),.cache/prefix_feature_cache_compare) \
	TG_PREFIX_FORCE_REBUILD=$(or $(TG_PREFIX_FORCE_REBUILD),false) \
	bash scripts/run_comparison.sh

compare-prefix-cold: ## compare-prefix after clearing the persistent prefix cache dir
	rm -rf $(or $(CACHE_DIR),.cache/prefix_feature_cache_compare)
	$(MAKE) compare-prefix \
		BUDGET=$(or $(BUDGET),240) \
		MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
		QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
		EVAL_POINTS=$(or $(EVAL_POINTS),3) \
		MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false) \
		CACHE_DIR=$(or $(CACHE_DIR),.cache/prefix_feature_cache_compare) \
		TG_PREFIX_FORCE_REBUILD=false

compare-prefix-warm: ## compare-prefix reusing the existing persistent prefix cache dir
	$(MAKE) compare-prefix \
		BUDGET=$(or $(BUDGET),240) \
		MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
		QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
		EVAL_POINTS=$(or $(EVAL_POINTS),3) \
		MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false) \
		CACHE_DIR=$(or $(CACHE_DIR),.cache/prefix_feature_cache_compare) \
		TG_PREFIX_FORCE_REBUILD=false

compare-prefix-coldwarm: ## Run compare-prefix twice: cold first, then warm with the same cache dir
	$(MAKE) compare-prefix-cold \
		BUDGET=$(or $(BUDGET),240) \
		MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
		QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
		EVAL_POINTS=$(or $(EVAL_POINTS),3) \
		MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false) \
		CACHE_DIR=$(or $(CACHE_DIR),.cache/prefix_feature_cache_compare)
	$(MAKE) compare-prefix-warm \
		BUDGET=$(or $(BUDGET),240) \
		MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
		QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
		EVAL_POINTS=$(or $(EVAL_POINTS),3) \
		MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false) \
		CACHE_DIR=$(or $(CACHE_DIR),.cache/prefix_feature_cache_compare)

paper-memory: ## Multi-seed paper memory suite: suffix-only baseline vs frozen prefix-cache TG-LoRA
	CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),) \
	TARGET_BP=$(or $(TARGET_BP),240) \
	MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
	QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
	EVAL_POINTS=$(or $(EVAL_POINTS),3) \
	SEEDS="$(or $(SEEDS),42 43 44)" \
	OUTPUT_BASE=$(or $(OUTPUT_BASE),runs/paper_memory_suite_$(shell date +%Y%m%d_%H%M%S)) \
	BASELINE_CONFIG=$(or $(BASELINE_CONFIG),configs/9b_baseline_suffix_only_last25.yaml) \
	TG_CONFIG=$(or $(TG_CONFIG),configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml) \
	CACHE_BASE=$(or $(CACHE_BASE),.cache/prefix_feature_cache_paper_suite) \
	MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false) \
	bash scripts/run_paper_memory_suite.sh

paper-memory-dry-run: ## Validate paper-memory suite config without GPU (dry-run mode)
	DRY_RUN=true \
	CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),) \
	TARGET_BP=$(or $(TARGET_BP),240) \
	MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
	QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
	EVAL_POINTS=$(or $(EVAL_POINTS),3) \
	SEEDS="$(or $(SEEDS),42 43 44)" \
	OUTPUT_BASE=$(or $(OUTPUT_BASE),runs/paper_memory_suite_dryrun_$(shell date +%Y%m%d_%H%M%S)) \
	BASELINE_CONFIG=$(or $(BASELINE_CONFIG),configs/9b_baseline_suffix_only_last25.yaml) \
	TG_CONFIG=$(or $(TG_CONFIG),configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml) \
	CACHE_BASE=$(or $(CACHE_BASE),.cache/prefix_feature_cache_paper_suite) \
	MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false) \
	bash scripts/run_paper_memory_suite.sh

paper-memory-one-shot: ## Multi-seed paper memory suite using one-shot SSD-backed prefix cache
	$(MAKE) paper-memory \
		CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),) \
		TARGET_BP=$(or $(TARGET_BP),240) \
		MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
		QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
		EVAL_POINTS=$(or $(EVAL_POINTS),3) \
		SEEDS="$(or $(SEEDS),42 43 44)" \
		OUTPUT_BASE=$(or $(OUTPUT_BASE),runs/paper_memory_one_shot_suite_$(shell date +%Y%m%d_%H%M%S)) \
		BASELINE_CONFIG=$(or $(BASELINE_CONFIG),configs/9b_baseline_suffix_only_last25.yaml) \
		TG_CONFIG=$(or $(TG_CONFIG),configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml) \
		CACHE_BASE=$(or $(CACHE_BASE),.cache/prefix_feature_cache_paper_suite_one_shot) \
		MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false)

paper-memory-cache-ablation: ## G4 ablation: isolate cache vs trajectory extrapolation (3-condition × 3-seed)
	bash scripts/run_ablation_cache_isolation.sh \
		$(if $(CUDA_VISIBLE_DEVICES),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)) \
		TARGET_BP=$(or $(TARGET_BP),240) \
		MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
		QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
		EVAL_POINTS=$(or $(EVAL_POINTS),3) \
		SEEDS="$(or $(SEEDS),42 43 44)" \
		OUTPUT_BASE=$(or $(OUTPUT_BASE),runs/ablation_cache_isolation_$(shell date +%Y%m%d_%H%M%S)) \
		EXISTING_TG_SUITE=$(or $(EXISTING_TG_SUITE),) \
		DRY_RUN=$(or $(DRY_RUN),false) \
		MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false)

cosine-n-ablation: ## Component 2 ablation: baseline vs fixed-N vs cosine-driven N (3-seed)
	CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),0) \
	TARGET_BP=$(or $(TARGET_BP),240) \
	MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
	QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
	EVAL_POINTS=$(or $(EVAL_POINTS),3) \
	SEEDS="$(or $(SEEDS),42 43 44)" \
	OUTPUT_BASE=$(or $(OUTPUT_BASE),runs/cosine_n_ablation_$(shell date +%Y%m%d_%H%M%S)) \
	BASELINE_CONFIG=$(or $(BASELINE_CONFIG),configs/9b_baseline_suffix_only_last25.yaml) \
	FIXED_CONFIG=$(or $(FIXED_CONFIG),configs/9b_tg_lora_fixed_n_persistent.yaml) \
	COSINE_CONFIG=$(or $(COSINE_CONFIG),configs/9b_tg_lora_cosine_n_persistent.yaml) \
	MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false) \
	SAVE_TRAJECTORY=$(or $(SAVE_TRAJECTORY),false) \
	CONFIDENT_SKIP_COS=$(or $(CONFIDENT_SKIP_COS),0.0) \
	CONFIDENT_SKIP_MIN_CYCLES=$(or $(CONFIDENT_SKIP_MIN_CYCLES),10) \
	ACCEPT_EVAL_EXAMPLES=$(ACCEPT_EVAL_EXAMPLES) \
	VALIDATION_SKIP_ENABLED=$(VALIDATION_SKIP_ENABLED) \
	VALIDATION_SKIP_HIGH_COS=$(VALIDATION_SKIP_HIGH_COS) \
	VALIDATION_SKIP_MID_COS=$(VALIDATION_SKIP_MID_COS) \
	VALIDATION_SKIP_MID_EVAL_EVERY=$(VALIDATION_SKIP_MID_EVAL_EVERY) \
	VALIDATION_SKIP_MIN_CYCLES=$(VALIDATION_SKIP_MIN_CYCLES) \
	VALIDATION_SKIP_FORCE_EVAL_N=$(VALIDATION_SKIP_FORCE_EVAL_N) \
	DRY_RUN=$(or $(DRY_RUN),false) \
	bash scripts/run_cosine_n_ablation.sh

cosine-n-ablation-dry-run: ## Validate cosine-N ablation configs without training
	$(MAKE) cosine-n-ablation DRY_RUN=true

cosine-n-skip-ablation: ## Component 2 ablation with probe eval and cosine-gated post-eval skip
	$(MAKE) cosine-n-ablation \
		COSINE_CONFIG=configs/9b_tg_lora_cosine_n_skip_persistent.yaml \
		OUTPUT_BASE=$(or $(OUTPUT_BASE),runs/cosine_n_skip_ablation_$(shell date +%Y%m%d_%H%M%S)) \
		ACCEPT_EVAL_EXAMPLES=$(or $(ACCEPT_EVAL_EXAMPLES),1) \
		VALIDATION_SKIP_ENABLED=$(or $(VALIDATION_SKIP_ENABLED),true) \
		VALIDATION_SKIP_HIGH_COS=$(or $(VALIDATION_SKIP_HIGH_COS),0.85) \
		VALIDATION_SKIP_MID_COS=$(or $(VALIDATION_SKIP_MID_COS),0.70) \
		VALIDATION_SKIP_MID_EVAL_EVERY=$(or $(VALIDATION_SKIP_MID_EVAL_EVERY),3) \
		VALIDATION_SKIP_MIN_CYCLES=$(or $(VALIDATION_SKIP_MIN_CYCLES),1) \
		VALIDATION_SKIP_FORCE_EVAL_N=$(or $(VALIDATION_SKIP_FORCE_EVAL_N),20)

cosine-n-skip-ablation-dry-run: ## Validate cosine-N skip ablation configs without training
	$(MAKE) cosine-n-skip-ablation DRY_RUN=true

paper-memory-compare-modes: ## Compare aggregate summaries from reuse vs one-shot paper-memory suites
	$(PYTHON_VENV) scripts/compare_paper_memory_modes.py \
		--reuse-summary $(REUSE_SUMMARY) \
		--one-shot-summary $(ONE_SHOT_SUMMARY) \
		$(if $(OUTPUT_BASE),--output-base $(OUTPUT_BASE))

paper-memory-all-modes: ## Run reuse suite, one-shot suite, then compare aggregate summaries
	$(MAKE) paper-memory \
		CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),) \
		TARGET_BP=$(or $(TARGET_BP),240) \
		MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
		QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
		EVAL_POINTS=$(or $(EVAL_POINTS),3) \
		SEEDS="$(or $(SEEDS),42 43 44)" \
		OUTPUT_BASE=$(or $(OUTPUT_ROOT),runs/paper_memory_modes_$(NOW_STAMP))/reuse \
		BASELINE_CONFIG=$(or $(BASELINE_CONFIG),configs/9b_baseline_suffix_only_last25.yaml) \
		TG_CONFIG=$(or $(REUSE_TG_CONFIG),configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml) \
		CACHE_BASE=$(or $(REUSE_CACHE_BASE),.cache/prefix_feature_cache_paper_suite) \
		MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false)
	$(MAKE) paper-memory-one-shot \
		CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),) \
		TARGET_BP=$(or $(TARGET_BP),240) \
		MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),1024) \
		QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
		EVAL_POINTS=$(or $(EVAL_POINTS),3) \
		SEEDS="$(or $(SEEDS),42 43 44)" \
		OUTPUT_BASE=$(or $(OUTPUT_ROOT),runs/paper_memory_modes_$(NOW_STAMP))/one_shot \
		BASELINE_CONFIG=$(or $(BASELINE_CONFIG),configs/9b_baseline_suffix_only_last25.yaml) \
		TG_CONFIG=$(or $(ONE_SHOT_TG_CONFIG),configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml) \
		CACHE_BASE=$(or $(ONE_SHOT_CACHE_BASE),.cache/prefix_feature_cache_paper_suite_one_shot) \
		MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false)
	$(MAKE) paper-memory-compare-modes \
		REUSE_SUMMARY=$(or $(OUTPUT_ROOT),runs/paper_memory_modes_$(NOW_STAMP))/reuse/aggregate_summary.json \
		ONE_SHOT_SUMMARY=$(or $(OUTPUT_ROOT),runs/paper_memory_modes_$(NOW_STAMP))/one_shot/aggregate_summary.json \
		OUTPUT_BASE=$(or $(COMPARE_OUTPUT_BASE),$(or $(OUTPUT_ROOT),runs/paper_memory_modes_$(NOW_STAMP))/mode_compare)

paper-memory-evaluate-gates: ## Evaluate paper gates G0–G4 against aggregate_summary.json
	$(PYTHON_VENV) scripts/evaluate_paper_gates.py \
		$(or $(GATE_SUMMARY),runs/paper_memory_suite_$(shell date +%Y%m%d*)/aggregate_summary.json) \
		$(if $(GATE_OUTPUT),-o $(GATE_OUTPUT)) \
		$(if $(GATE_SKIP),--skip-gates $(GATE_SKIP)) \
		$(if $(G1_LOSS_RED_RATIO),--g1-loss-red-ratio $(G1_LOSS_RED_RATIO)) \
		$(if $(G1_QUALITY_TOLERANCE),--g1-quality-tolerance $(G1_QUALITY_TOLERANCE)) \
		$(if $(G2_MEMORY_IMPROVEMENT),--g2-memory-improvement $(G2_MEMORY_IMPROVEMENT)) \
		$(if $(FRONTIER_REPORT),--frontier-report $(FRONTIER_REPORT)) \
		$(if $(COLD_SUMMARY),--cold-summary $(COLD_SUMMARY)) \
		$(if $(NO_CACHE_SUMMARY),--no-cache-summary $(NO_CACHE_SUMMARY))

paper-memory-external-eval: ## Run external quality evaluation (G3 Gate) on best models from paper-memory suite
	$(PYTHON_VENV) scripts/run_paper_external_eval.py \
		$(or $(GATE_SUMMARY),runs/paper_memory_suite_$(shell date +%Y%m%d*)/aggregate_summary.json) \
		$(if $(EXTERNAL_EVAL),--external-eval $(EXTERNAL_EVAL)) \
		$(if $(EVAL_TASKS),--tasks $(EVAL_TASKS)) \
		$(if $(EVAL_OUTPUT),--output $(EVAL_OUTPUT)) \
		$(if $(EVAL_BATCH_SIZE),--batch-size $(EVAL_BATCH_SIZE)) \
		$(if $(EVAL_LIMIT),--limit $(EVAL_LIMIT))

paper-memory-frontier-sweep: ## Stage 3 frontier sweep: run paper-memory at increasing MAX_SEQ_LEN, detect frontier separation
	CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),) \
	TARGET_BP=$(or $(TARGET_BP),240) \
	SEEDS="$(or $(SEEDS),42 43 44)" \
	SEQS="$(or $(SEQS),1536 2048 3072)" \
	QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),32) \
	EVAL_POINTS=$(or $(EVAL_POINTS),3) \
	OUTPUT_BASE=$(or $(OUTPUT_BASE),runs/frontier_sweep_$(NOW_STAMP)) \
	BASELINE_CONFIG=$(or $(BASELINE_CONFIG),configs/9b_baseline_suffix_only_last25.yaml) \
	TG_CONFIG=$(or $(TG_CONFIG),configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml) \
	CACHE_BASE=$(or $(CACHE_BASE),.cache/prefix_feature_cache_frontier) \
	MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false) \
	bash scripts/run_frontier_sweep.sh

precompute-prefix-cache: ## Offline multi-GPU prefix-cache precompute for the frozen paper config
	$(PYTHON_VENV) scripts/precompute_prefix_cache_parallel.py \
		--config $(or $(PREFIX_CACHE_CONFIG),configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml) \
		--datasets $(or $(DATASETS),auto) \
		$(if $(DEVICES),--devices $(DEVICES)) \
		$(if $(CACHE_DIR),--cache-dir $(CACHE_DIR)) \
		$(if $(SUMMARY_PATH),--summary-path $(SUMMARY_PATH)) \
		$(if $(FORCE_REBUILD),--force-rebuild) \
		$(if $(KEEP_SHARDS),--keep-shards)

compare-report: ## Generate comparison report from existing runs
	$(PYTHON_VENV) scripts/compare_runs.py \
		--baseline $(BASELINE_METRICS) \
		--tg-lora $(TG_LORA_METRICS) \
		--output-dir reports

ablation: ## Run full ablation suite (4 configs, TARGET_BP=backward passes)
	TARGET_BP=$(or $(TARGET_BP),3600) bash scripts/run_ablation_suite.sh

sweep: ## Run TG-LoRA hyperparameter sweep
	SWEEP_BUDGET=$(or $(SWEEP_BUDGET),200) bash scripts/run_sweep.sh

accel-sweep: ## Run accel param sweep (4 configs: decay x boost grid)
	bash scripts/run_accel_sweep.sh

# ── Benchmarks ───────────────────────────────────────────────────────────────

bench-optimizer: ## Benchmark optimizer recreate vs state-reuse (~5 min)
	$(PYTHON_VENV) scripts/benchmark_optimizer_lifecycle.py --config $(CONFIG)

bench-prefix-cache: ## Benchmark persistent prefix cache cold vs warm reuse
	$(PYTHON_VENV) scripts/benchmark_prefix_cache.py \
		--budget $(or $(BUDGET),32) \
		--max-seq-len $(or $(MAX_SEQ_LEN),256) \
		--quick-eval-examples $(or $(QUICK_EVAL_EXAMPLES),4) \
		--eval-points $(or $(EVAL_POINTS),2) \
		--baseline-config $(or $(BASELINE_CONFIG),configs/9b_baseline_suffix_only_last25.yaml) \
		--tg-config $(or $(TG_CONFIG),configs/9b_tg_lora_prefix_feature_cache_experimental.yaml) \
		--cache-dir $(or $(CACHE_DIR),.cache/prefix_feature_cache_benchmark) \
		$(if $(CUDA_VISIBLE_DEVICES),--cuda-visible-devices $(CUDA_VISIBLE_DEVICES)) \
		$(if $(OUTPUT_BASE),--output-base $(OUTPUT_BASE)) \
		$(if $(filter true,$(MLFLOW_ENABLED)),--mlflow-enabled)

bench-prefix-cache-one-shot: ## Benchmark prefix cache in one-shot SSD-backed mode
	$(MAKE) bench-prefix-cache \
		BUDGET=$(or $(BUDGET),32) \
		MAX_SEQ_LEN=$(or $(MAX_SEQ_LEN),256) \
		QUICK_EVAL_EXAMPLES=$(or $(QUICK_EVAL_EXAMPLES),4) \
		EVAL_POINTS=$(or $(EVAL_POINTS),2) \
		BASELINE_CONFIG=$(or $(BASELINE_CONFIG),configs/9b_baseline_suffix_only_last25.yaml) \
		TG_CONFIG=$(or $(TG_CONFIG),configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml) \
		CACHE_DIR=$(or $(CACHE_DIR),.cache/prefix_feature_cache_one_shot_benchmark) \
		OUTPUT_BASE=$(or $(OUTPUT_BASE),runs/prefix_cache_one_shot_benchmark_$(shell date +%Y%m%d_%H%M%S)) \
		MLFLOW_ENABLED=$(or $(MLFLOW_ENABLED),false)

analyze-prefix-break-even: ## Quantify when cold-build cost amortizes against warm TG wall-clock savings
	$(PYTHON_VENV) scripts/analyze_prefix_cache_break_even.py \
		--paper-summary $(PAPER_SUMMARY) \
		$(if $(PRECOMPUTE_SUMMARY),--precompute-summary $(PRECOMPUTE_SUMMARY)) \
		$(if $(OUTPUT_PATH),--output $(OUTPUT_PATH))

bench-velocity-ops: ## Benchmark velocity EMA update and cap_update in-place ops
	$(PYTHON_VENV) scripts/benchmark_velocity_ops.py \
		--iterations $(or $(ITERATIONS),1000)

bench-velocity-ops-ci: ## Run velocity ops benchmark against checked-in baseline (CI gate)
	$(PYTHON_VENV) scripts/benchmark_velocity_ops.py \
		--quick --baseline baselines/velocity_ops.json --threshold 20

bench-velocity-ops-save-baseline: ## Regenerate baselines/velocity_ops.json (commit after running)
	@mkdir -p baselines
	$(PYTHON_VENV) scripts/benchmark_velocity_ops.py \
		--quick --save-baseline baselines/velocity_ops.json

# ── Quality ──────────────────────────────────────────────────────────────────

test: ## Run unit tests
	$(PYTHON_VENV) -m pytest tests/ -v

test-accel: ## Run accel sweep pipeline tests (92 tests across 4 files)
	$(PYTHON_VENV) -m pytest tests/ -k "accel" -v

test-cov: ## Run tests with coverage
	$(PYTHON_VENV) -m pytest tests/ --cov=src --cov-report=term-missing

test-integration: ## Run E2E integration tests only
	$(PYTHON_VENV) -m pytest tests/test_advise_training_e2e.py tests/test_analyze_trajectory_e2e.py -v

test-trajectory: ## Run Phase 59-61 trajectory/advisor tests
	$(PYTHON_VENV) -m pytest tests/test_trajectory.py tests/test_trajectory_controller.py tests/test_training_advisor.py tests/test_analyze_trajectory_e2e.py tests/test_advise_training_e2e.py -v

test-cli-help: ## Verify all Python CLI scripts respond to --help
	$(PYTHON_VENV) -m pytest tests/test_cli_help_smoke.py -v

check-spine: ## Verify spec/doc spine-anchor integrity (no provenance drift)
	$(PYTHON_VENV) scripts/check_spine_anchors.py

lint: ## Run linting
	$(PYTHON_VENV) -m ruff check src/ tests/ scripts/
	$(PYTHON_VENV) -m ruff format --check src/ tests/ scripts/

format: ## Auto-format code
	$(PYTHON_VENV) -m ruff format src/ tests/ scripts/

clean: ## Clean generated files
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-data: ## Remove downloaded/generated data (careful!)
	rm -rf data/raw data/train.jsonl data/valid_*.jsonl data/gold_test.jsonl

clean-runs: ## Remove all experiment runs (careful!)
	rm -rf runs/*

# ── Operations ────────────────────────────────────────────────────────────────

check-status: ## Run agent autonomy status check to find next steps
	chmod +x scripts/agent_check_status.py
	$(PYTHON_VENV) scripts/agent_check_status.py

diagnose: ## Run health check on GPU, checkpoint, config, or logs
	$(PYTHON_VENV) scripts/diagnose.py $(ARGS)

recover: ## Automated fault recovery from interrupted training
	$(PYTHON_VENV) scripts/recover.py $(ARGS)

# ── MLX (Apple Silicon native 4-bit) ─────────────────────────────────────────

MLX_MODEL ?= .cache/mlx_models/$(shell echo $(BASE_MODEL) | sed 's/\//--/g')
MLX_DATA ?= data_mlx
MLX_ITERS ?= 1500
MLX_MAX_SEQ_LENGTH ?= 2048
MLX_MAX_OPS_PER_BUFFER ?= 4
MLX_MAX_MB_PER_BUFFER ?= 32
MLX_BFS_MAX_WIDTH ?= 4
MLX_GATED_DELTA_CHUNK ?= 512
MLX_CLEAR_CACHE_THRESHOLD ?= 8589934592

convert-mlx: ## Convert HF model to MLX 4-bit (run once per model)
	@mkdir -p $(dir $(MLX_MODEL))
	$(PYTHON_VENV) -c "from mlx_lm.convert import convert; convert('$(BASE_MODEL)', '$(MLX_MODEL)', quantize=True, q_group_size=64, q_bits=4)"
	@echo "Done: $(MLX_MODEL)"

convert-mlx-path: ## Show resolved MLX model path
	@echo $(MLX_MODEL)

mlx-data: ## Symlink training data for MLX-LM (train.jsonl + valid.jsonl)
	@mkdir -p $(MLX_DATA)
	@test -f data/train.jsonl || (echo "Run 'make download-data && make prepare-data' first" && exit 1)
	@ln -sf $(shell pwd)/data/train.jsonl $(MLX_DATA)/train.jsonl
	@ln -sf $(shell pwd)/data/valid_quick.jsonl $(MLX_DATA)/valid.jsonl
	@echo "Done: $(MLX_DATA)/  (train=$(shell wc -l < data/train.jsonl) valid=$(shell wc -l < data/valid_quick.jsonl))"

train-mlx: ## Run MLX QLoRA with bounded Metal resource lifetime
	MLX_MAX_OPS_PER_BUFFER=$(MLX_MAX_OPS_PER_BUFFER) \
	MLX_MAX_MB_PER_BUFFER=$(MLX_MAX_MB_PER_BUFFER) \
	MLX_BFS_MAX_WIDTH=$(MLX_BFS_MAX_WIDTH) \
	MLX_GATED_DELTA_CHUNK=$(MLX_GATED_DELTA_CHUNK) \
	$(PYTHON_VENV) mlx/scripts/train_lora_fixed.py \
		--model $(MLX_MODEL) \
		--data $(MLX_DATA) \
		--train \
		--iters $(MLX_ITERS) \
		--num-layers 32 \
		--batch-size 1 \
		--grad-accumulation-steps 8 \
		--learning-rate 2e-4 \
		--max-seq-length $(MLX_MAX_SEQ_LENGTH) \
		--clear-cache-threshold $(MLX_CLEAR_CACHE_THRESHOLD) \
		--wired-limit-ratio 0.8 \
		--memory-limit-ratio 0.8 \
		--cache-limit-ratio 0.15 \
		--steps-per-report 10 \
		--steps-per-eval 250 \
		--save-every 250 \
		--adapter-path runs/mlx_qlora_$(NOW_STAMP) \
		--seed 42

train-mlx-continuous: ## Run one continuous MLX process with bounded Metal scheduling
	$(MAKE) train-mlx MLX_ITERS=$(MLX_ITERS)

train-mlx-upstream: ## Run upstream mlx_lm.lora path for reproducing Metal OOM
	$(PYTHON_VENV) mlx/scripts/run_lora_guarded.py \
		--model $(MLX_MODEL) \
		--data $(MLX_DATA) \
		--train \
		--iters $(MLX_ITERS) \
		--num-layers 32 \
		--batch-size 1 \
		--grad-accumulation-steps 8 \
		--learning-rate 2e-4 \
		--max-seq-length $(MLX_MAX_SEQ_LENGTH) \
		--clear-cache-threshold 1 \
		--wired-limit-ratio 0.8 \
		--memory-limit-ratio 0.8 \
		--cache-limit-ratio 0 \
		--steps-per-report 10 \
		--steps-per-eval 250 \
		--save-every 250 \
		--adapter-path runs/mlx_qlora_continuous_$(NOW_STAMP) \
		--seed 42

train-mlx-smoke: ## Quick MLX smoke test (20 steps, ~30s)
	MLX_MAX_OPS_PER_BUFFER=$(MLX_MAX_OPS_PER_BUFFER) \
	MLX_MAX_MB_PER_BUFFER=$(MLX_MAX_MB_PER_BUFFER) \
	MLX_BFS_MAX_WIDTH=$(MLX_BFS_MAX_WIDTH) \
	MLX_GATED_DELTA_CHUNK=$(MLX_GATED_DELTA_CHUNK) \
	$(PYTHON_VENV) mlx/scripts/train_lora_fixed.py \
		--model $(MLX_MODEL) \
		--data $(MLX_DATA) \
		--train \
		--iters 20 \
		--num-layers 32 \
		--batch-size 1 \
		--grad-accumulation-steps 8 \
		--learning-rate 2e-4 \
		--max-seq-length $(MLX_MAX_SEQ_LENGTH) \
		--clear-cache-threshold $(MLX_CLEAR_CACHE_THRESHOLD) \
		--wired-limit-ratio 0.8 \
		--memory-limit-ratio 0.8 \
		--cache-limit-ratio 0.15 \
		--steps-per-report 1 \
		--steps-per-eval 10 \
		--adapter-path runs/mlx_smoke_$(NOW_STAMP) \
		--seed 42

MLX_BASELINE_CONFIG ?= configs/mlx_baseline_500.yaml

train-mlx-baseline: ## Run MLX baseline training with auto-resume (uses config file). MLX_BASELINE_CONFIG to override config.
	MLX_MAX_OPS_PER_BUFFER=$(MLX_MAX_OPS_PER_BUFFER) \
	MLX_MAX_MB_PER_BUFFER=$(MLX_MAX_MB_PER_BUFFER) \
	MLX_BFS_MAX_WIDTH=$(MLX_BFS_MAX_WIDTH) \
	MLX_GATED_DELTA_CHUNK=$(MLX_GATED_DELTA_CHUNK) \
	$(PYTHON_VENV) mlx/scripts/train_lora_fixed.py \
		--config $(MLX_BASELINE_CONFIG)

compare-mlx: ## Compare MPS baseline vs MLX baseline
	@if [ -z "$(BASELINE_RUN)" ] || [ -z "$(MLX_RUN)" ]; then \
		echo "Usage: make compare-mlx BASELINE_RUN=runs/<mps-run> MLX_RUN=runs/<mlx-run>"; \
		exit 1; \
	fi
	$(PYTHON_VENV) scripts/compare_runs.py \
		--baseline $(BASELINE_RUN) \
		--tg-lora $(MLX_RUN) \
		--output-dir reports

convert-mlx-35b: ## Convert HF 35B model to MLX 4-bit (run once)
	@mkdir -p $(dir $(MODEL_35B))
	$(PYTHON_VENV) -c "from mlx_lm.convert import convert; convert('Qwen/Qwen3.6-35B-A3B', '$(MODEL_35B)', quantize=True, q_group_size=64, q_bits=4)"
	@echo "Done: $(MODEL_35B)"

train-mlx-35b: ## Run Qwen3.6-35B-A3B QLoRA training on MLX (500 steps)
	$(MAKE) train-mlx MLX_MODEL=$(MODEL_35B) MLX_ITERS=500

sync-to-cuda: ## Rsync MLX runs from Mac to CUDA host. Usage: make sync-to-cuda HOST=user@cuda-host DEST=/path/to/tg-lora/runs
	@if [ -z "$(HOST)" ] || [ -z "$(DEST)" ]; then \
		echo "Usage: make sync-to-cuda HOST=user@cuda-host DEST=/path/to/tg-lora/runs"; \
		exit 1; \
	fi
	rsync -avz runs/mlx_* $(HOST):$(DEST)/

# ── CI ────────────────────────────────────────────────────────────────────────

ci: ## Run full CI pipeline (lint + test + script import check)
	$(PYTHON_VENV) -m ruff check src/ tests/ scripts/ mlx/
	$(PYTHON_VENV) -m ruff format --check src/ tests/ scripts/ mlx/
	$(PYTHON_VENV) scripts/check_spine_anchors.py
	$(PYTHON_VENV) -m pytest tests/ mlx/tests/ -q
	@$(PYTHON_VENV) -m pytest tests/ -k "accel" -q --tb=short
	@$(PYTHON_VENV) -c "import scripts.diagnose; import scripts.recover; print('scripts import OK')" 2>/dev/null || \
		$(PYTHON_VENV) -c "import importlib.util; \
		util=importlib.util.spec_from_file_location('diagnose','scripts/diagnose.py'); \
		mod=importlib.util.module_from_spec(util); util.loader.exec_module(mod); \
		util2=importlib.util.spec_from_file_location('recover','scripts/recover.py'); \
		mod2=importlib.util.module_from_spec(util2); util2.loader.exec_module(mod2); \
		print('script imports OK')"

# ── Docker Orchestration ──────────────────────────────────────────────────────

docker-build: ## Build the docker image locally
	docker compose build

docker-test: ## Run tests inside the docker container
	docker compose run --rm tg-lora pytest tests/ -v

docker-run: ## Start an interactive bash session inside the docker container with GPU access
	docker compose run --rm --entrypoint bash tg-lora

docker-eval: ## Run the 3-seed downstream evaluation inside the docker container
	docker compose run --rm --entrypoint python tg-lora scripts/run_all_seeds_eval.py

# ── Evidence Management ────────────────────────────────────────────────────────

PLAN_CONFIG ?= configs/paper_experiment_plan.yaml

run-paper-experiment: ## Run paper experiment plan end-to-end
	chmod +x scripts/run_experiment_plan.py
	$(PYTHON_VENV) scripts/run_experiment_plan.py --config $(PLAN_CONFIG)

ingest-evidence: ## Automatically discover and ingest paper-flagged runs to paper_evidence/
	$(PYTHON_VENV) scripts/ingest_paper_evidence.py

check-evidence: ## Validate paper_evidence size and check for binary files and metadata
	@echo "=== Checking paper_evidence size ==="
	@du -sh paper_evidence/
	@echo "=== Checking for unauthorized binary checkpoint files ==="
	@LARGE_FILES=$$(find paper_evidence/ -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" -o -name "*.pth" -o -name "*.gguf" \)); \
	if [ -n "$$LARGE_FILES" ]; then \
		echo "ERROR: Large binaries found in paper_evidence/:"; \
		echo "$$LARGE_FILES"; \
		exit 1; \
	else \
		echo "[OK] No heavy binaries detected."; \
	fi
	@echo "=== Checking for version_metadata.json ==="
	@MISSING_META=$$(find paper_evidence/ -mindepth 1 -maxdepth 1 -type d ! -exec test -e "{}/version_metadata.json" \; -print); \
	if [ -n "$$MISSING_META" ]; then \
		echo "WARNING: Missing version_metadata.json in some evidence directories:"; \
		echo "$$MISSING_META"; \
	else \
		echo "[OK] All evidence directories have version_metadata.json."; \
	fi

# ── Progressive Freezing research tooling (Category-C attack) ────────────────
#
# The GOAL §4 valid_loss-axis significance verdict, from a REAL run. Trains the
# progressive-freeze trio on a small learnable proxy for an output-first
# candidate vs random-order surrogates across seeds and feeds the resulting real
# valid_loss samples through surrogate_valid_loss_ci() — the first significance
# verdict grounded in numbers from an actual run (proxy-scale; target run swaps
# the data source through the same function). Auto-selects CUDA when available.
# Needs a torch-enabled interpreter: PYTHON_VENV=/path/to/torch-python make ...
FREEZE_VALIDLOSS_CI_FLAGS ?= --device auto

freeze-validloss-ci: ## GOAL §4 real valid_loss-axis significance run (proxy-scale; auto CUDA)
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci $(FREEZE_VALIDLOSS_CI_FLAGS)

freeze-validloss-ci-heterogeneous: ## Positive control: heterogeneous (per-layer rank) stack, auto CUDA
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci $(FREEZE_VALIDLOSS_CI_FLAGS) --architecture heterogeneous

freeze-validloss-ci-generalize: ## Conclusive-TIES run: held-out generalization task, auto CUDA
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci $(FREEZE_VALIDLOSS_CI_FLAGS) --task generalize

# The apparatus order-resolution diagnostic. Variance-decomposes the proxy's
# final valid_loss into a Var(order) signal (distinct freeze orders at a fixed
# seed) vs a Var(seed) noise floor (a fixed order across seeds) and reports their
# ratio — does the §4 verdict apparatus resolve freeze order at proxy scale?
# The proxy finding is ratio=0.000 (order is unresolvable: the boundary local
# loss does not couple to the held-out task metric), which makes the verdict TIES
# a genuine null and target-scale proven necessary. Auto-selects CUDA.
# Needs a torch-enabled interpreter: PYTHON_VENV=/path/to/torch-python make ...
FREEZE_ORDER_SENSITIVITY_FLAGS ?= --device auto

freeze-order-sensitivity: ## Order-resolution diagnostic: can the proxy resolve freeze order? (auto CUDA)
	$(PYTHON_VENV) -m scripts.run_freeze_order_sensitivity $(FREEZE_ORDER_SENSITIVITY_FLAGS)

