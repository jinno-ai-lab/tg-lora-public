.PHONY: setup install inspect download-data prepare-data \
       smoke smoke-tg smoke-bl \
       train-baseline train-tg-lora train-tg-lora-optreuse train-tg-lora-prefix \
       eval eval-lora eval-downstream eval-downstream-mlx eval-llm-jp-eval-mlx eval-mlx eval-base eval-35b-base eval-35b-ckpt \
       ingest-evidence check-evidence run-paper-experiment freeze-validloss-ci freeze-validloss-ci-heterogeneous freeze-validloss-ci-generalize freeze-validloss-ci-9b freeze-validloss-ci-9b-generalization freeze-validloss-ci-9b-heterogeneous-generalization freeze-validloss-ci-9b-baseline freeze-validloss-ci-9b-full freeze-validloss-ci-9b-full-bg freeze-validloss-ci-9b-full-heterogeneous freeze-validloss-ci-9b-full-heterogeneous-bg freeze-replay \

	compare compare-prefix compare-prefix-cold compare-prefix-warm compare-prefix-coldwarm compare-report paper-memory paper-memory-dry-run paper-memory-one-shot paper-memory-compare-modes paper-memory-all-modes paper-memory-evaluate-gates paper-memory-external-eval paper-memory-frontier-sweep paper-memory-cache-ablation cosine-n-ablation cosine-n-ablation-dry-run cosine-n-skip-ablation cosine-n-skip-ablation-dry-run precompute-prefix-cache ablation sweep accel-sweep \
	bench-optimizer bench-prefix-cache bench-prefix-cache-one-shot analyze-prefix-break-even analyze-prefix-break-even-ci bench-velocity-ops bench-velocity-ops-ci bench-velocity-ops-save-baseline \
       test test-accel test-cov test-integration test-trajectory test-cli-help lint format clean clean-data clean-runs \
       diagnose recover ci gates-ci check-status \
       convert-mlx train-mlx train-mlx-baseline train-mlx-continuous train-mlx-upstream train-mlx-smoke mlx-data compare-mlx \
       help

PYTHON ?= python
VENV ?= .venv
PIP := $(VENV)/bin/pip
# `?=` (not `:=`) on purpose: the GPU targets are driven as
# `PYTHON_VENV=/path/to/torch-python make <target>` (env-var form, see the 8
# "Needs a torch+bnb+GPU interpreter" comments below) and a makefile `:=` would
# silently clobber that override back to .venv/bin/python (exit 127 when no
# .venv exists, e.g. in a worktree). `?=` lets an env/command-line PYTHON_VENV
# win while still defaulting to $(VENV)/bin/python. Verify any change with both
# `PYTHON_VENV=/x make -n <tgt>` and plain `make -n <tgt>`.
PYTHON_VENV ?= $(VENV)/bin/python
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

# (PROBE_* vars avoid colliding with the global CONFIG / SEQS. NB: defaults use
# ?= rather than $(or ...) because the seq-len list contains commas, which $(or)
# would split into separate arguments — collapsing the default to its first int.)
PROBE_CONFIG ?= configs/9b_baseline_suffix_only_last25.yaml
PROBE_SEQ_LENS ?= 256,512,768,1024,1280,1536,2048
PROBE_BATCH_SIZE ?= 1

probe-9b-memory: ## Empirical 9B suffix-only GPU memory-frontier probe (real Qwen3.5-9B; data-independent)
# Needs a torch+bnb+GPU interpreter: PYTHON_VENV=/path/to/torch-python make probe-9b-memory
# Binds configs/9b_baseline_suffix_only_last25.yaml to the GPU and measures peak
# per-step VRAM across seq_len via the trainer's exact public model-loading path
# (load_base_model -> apply_lora -> configure_trainable_lora_scope), bypassing
# only the private src.data blocker with a synthetic batch. Resolves whether the
# suffix-only config's memory stack fits seq>=1024 on 12GB (data-independent).
	$(PYTHON_VENV) -m scripts.probe_9b_memory_frontier \
		--config $(PROBE_CONFIG) \
		--seq-lens "$(PROBE_SEQ_LENS)" \
		--batch-size $(PROBE_BATCH_SIZE) \
		$(if $(PROBE_OUTPUT),--output $(PROBE_OUTPUT)) \
		$(if $(PROBE_JSON),--json)

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

analyze-prefix-break-even-ci: ## CI gate variant: fail (non-zero) unless the cold build amortizes per the gate vars
	$(PYTHON_VENV) scripts/analyze_prefix_cache_break_even.py \
		--paper-summary $(PAPER_SUMMARY) \
		$(if $(PRECOMPUTE_SUMMARY),--precompute-summary $(PRECOMPUTE_SUMMARY)) \
		$(if $(OUTPUT_PATH),--output $(OUTPUT_PATH)) \
		$(if $(REQUIRE_WARM_WIN),--require-warm-win) \
		$(if $(MAX_BREAK_EVEN_RUNS),--max-break-even-runs $(MAX_BREAK_EVEN_RUNS)) \
		$(if $(REQUIRE_ONE_RUN_WIN),--require-one-run-win) \
		$(if $(MAX_WARM_GPU_PEAK_MB),--max-warm-gpu-peak-mb $(MAX_WARM_GPU_PEAK_MB))

bench-velocity-ops: ## Benchmark velocity EMA update and cap_update in-place ops
	$(PYTHON_VENV) scripts/benchmark_velocity_ops.py \
		--iterations $(or $(ITERATIONS),1000)

bench-velocity-ops-ci: ## Portable CI gate: cap_update overhead ratio must stay within a hardware-normalized ceiling
	$(PYTHON_VENV) scripts/benchmark_velocity_ops.py \
		--quick --max-cap-overhead-ratio $(or $(MAX_CAP_OVERHEAD_RATIO),3.0)

bench-velocity-ops-save-baseline: ## Regenerate baselines/velocity_ops.json (commit after running)
	@mkdir -p baselines
	$(PYTHON_VENV) scripts/benchmark_velocity_ops.py \
		--quick --save-baseline baselines/velocity_ops.json

gates-ci: ## Run every GPU-free CI gate in one target (the loop's gate sequence). No GPU run needed.
	$(MAKE) bench-velocity-ops-ci
	# PAPER_SUMMARY / OUTPUT_PATH are overridable so the loop's gate sequence can
	# run against REAL pipeline output (a genuine GPU A/B summary), not only the
	# checked-in canonical fixture — closing the fixture-vs-pipeline gap at the
	# sequence level. Defaults preserve the canonical-fixture behavior (AI_HUB
	# feedback: confirm the gates are exercisable against real producer output).
	PAPER_SUMMARY=$(or $(PAPER_SUMMARY),tests/fixtures/prefix_break_even_canonical_summary.json) \
	REQUIRE_WARM_WIN=1 MAX_WARM_GPU_PEAK_MB=12288 \
	OUTPUT_PATH=$(or $(OUTPUT_PATH),runs/gates_ci/break_even_verdict.json) \
	$(MAKE) analyze-prefix-break-even-ci

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

ci: ## Run full CI pipeline (lint + gates + test + script import check)
	$(PYTHON_VENV) -m ruff check src/ tests/ scripts/ mlx/
	$(PYTHON_VENV) -m ruff format --check src/ tests/ scripts/ mlx/
	$(PYTHON_VENV) scripts/check_spine_anchors.py
	$(MAKE) gates-ci
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

freeze-validloss-ci-heterogeneous-generalize: ## Discriminating positive control: heterogeneous stack x generalize task (auto CUDA)
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci $(FREEZE_VALIDLOSS_CI_FLAGS) --architecture heterogeneous --task generalize

freeze-validloss-ci-heterogeneous-generalize-thin: ## Single-cycle guard: n=2/arm thin run on the discriminating leg (auto CUDA)
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci $(FREEZE_VALIDLOSS_CI_FLAGS) --architecture heterogeneous --task generalize --n-candidate 2 --n-surrogate 2

freeze-validloss-ci-negative-control: ## Sensitivity negative control: under-trained candidate fires real UNDERSHOOTS (auto CUDA)
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci $(FREEZE_VALIDLOSS_CI_FLAGS) --candidate-total 2 --json --output tests/fixtures/freeze_validloss_negative_control_proxy.json

freeze-validloss-ci-negative-control-surrogate: ## Symmetric sensitivity negative control: under-trained surrogate fires real SURPASSES (auto CUDA)
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci $(FREEZE_VALIDLOSS_CI_FLAGS) --surrogate-total 2 --json --output tests/fixtures/freeze_validloss_negative_control_surrogate_proxy.json

# The FIRST real-9B + real-data §4 A/B verdict. Binds the suffix-only config
# (configs/9b_baseline_suffix_only_last25.yaml) to the GPU and runs the real
# Qwen3.5-9B QLoRA — an output-first progressive-freeze candidate vs random-order
# surrogates — on real public Dolly data, feeding the real valid_loss samples to
# surrogate_valid_loss_ci with proxy_scale=False (target scale). The 9B model is
# loaded ONCE and each arm resets the LoRA adapter in place (a per-arm reload
# leaks a second ~5.5GB model and OOMs a 12GB GPU). seq_len defaults to 1024 (the
# suffix-only per-step fit point); override FREEZE_9B_FLAGS for a smaller smoke.
# 4 seeds/arm clears the is_thin_evidence bar (>=3); the deposit is honestly
# reduced-budget (short training, 20 steps << max_steps=1500) — a target-scale
# data point, NOT yet the full max_steps=1500 multi-seed §4 verdict.
# Needs a torch+bnb+GPU interpreter: PYTHON_VENV=/path/to/torch-python make freeze-validloss-ci-9b
FREEZE_9B_FLAGS ?= --seq-len 1024 --total-steps 20 --warmup-steps 6 --depth 3 --spacing 4 --n-candidate 4 --n-surrogate 4 --train-examples 8 --valid-examples 10 --max-dataset-rows 400

freeze-validloss-ci-9b: ## GOAL §4 REAL 9B target-scale A/B verdict (suffix-only, Dolly; auto CUDA)
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci_9b $(FREEZE_9B_FLAGS) --json --output tests/fixtures/freeze_validloss_ci_9b_surrogate.json

# GENERALIZATION-REGIME robustness arm of the real 9B §4 A/B. The default
# freeze-validloss-ci-9b above runs in a MEMORIZATION regime — 8 train examples
# cycled over 20 steps, so the LoRA drives train CE to ~0 (verified in its
# deposit provenance: candidate last_train_loss≈5e-4, control = 0.0). The
# held-out valid_loss is then dominated by the frozen base, barely perturbed by
# an overfit adapter — so the headline SURPASSES is measured on a model that has
# memorized, not generalized. This target re-runs the SAME A/B (candidate vs
# surrogate vs direction-control) in a GENERALIZATION regime: 48 train examples
# over 96 steps (2 epochs — each example seen ~2x, realistic SFT, train CE stays
# well above 0), 3 seeds/arm + 3-arm direction isolation. The deposit carries
# final_ce_train_loss per arm (mean full-CE over the train set) so the
# memorization-vs-generalization regime is machine-readable, not asserted. The
# research question: does the SURPASSES verdict SURVIVE moving out of the
# memorization regime, or was it a memorization artifact? Still reduced-budget
# (96 < max_steps=1500) and honest about it — this is a REGIME-ROBUSTNESS data
# point, NOT the full §4 verdict (which needs private src.data + >12GB).
# Needs a torch+bnb+GPU interpreter: PYTHON_VENV=/path/to/torch-python make freeze-validloss-ci-9b-generalization
FREEZE_9B_GEN_FLAGS ?= --seq-len 1024 --total-steps 96 --warmup-steps 12 --depth 3 --spacing 10 --n-candidate 3 --n-surrogate 3 --n-control 3 --train-examples 48 --valid-examples 32 --max-dataset-rows 600

freeze-validloss-ci-9b-generalization: ## GOAL §4 real-9B A/B in a GENERALIZATION regime (2-epoch; memorization-robustness arm)
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci_9b $(FREEZE_9B_GEN_FLAGS) --json --output tests/fixtures/freeze_validloss_ci_9b_generalization.json

# The HETEROGENEOUS (per-layer asymmetric rank) arm of the real-9B §4 A/B — the
# one §4 research task that was open at target scale. Every committed 9B verdict
# above runs on a HOMOGENEOUS LoRA (every active layer shares cfg.lora.r=16); the
# open question is whether the surrogate/direction/baseline verdicts hold when the
# adapter itself is asymmetric — output-side layers given more CAPACITY (higher
# rank) than input-side ones, via peft rank_pattern + alpha_pattern (alpha=2*rank
# held constant so only capacity, not magnitude, varies). This target re-runs the
# SAME generalization-regime A/B (FREEZE_9B_GEN_FLAGS — candidate output-first
# freeze vs random surrogate vs input-contig direction control, 48 train / 96 step
# / 2 epoch, 3 seeds/arm) with --architecture heterogeneous, so the deposit is a
# clean apples-to-apples comparison with freeze_validloss_ci_9b_generalization.json
# (architecture is the ONLY difference). Under heterogeneous ranks the candidate's
# output-first freeze naturally targets the HIGHEST-rank (highest-capacity) active
# layers — the architectural interaction this leg exists to probe. Still
# reduced-budget (96 < max_steps=1500) and honest about it: a target-scale
# heterogeneous data point, NOT the full §4 verdict.
# Needs a torch+bnb+GPU interpreter: PYTHON_VENV=/path/to/torch-python make freeze-validloss-ci-9b-heterogeneous-generalization
# The run-log witness is written BESIDE the deposit under tests/fixtures/ (a
# COMMITTED, tracked artifact) — not the gitignored runs/ default — so the
# verdict is independently reproducible without a GPU re-run: a skeptic fetches
# the committed loss-curve, recomputes run_log_sha256, and confirms the dynamics
# behind the verdict are the recorded ones (the content-hash axis of iter
# `65073bb`, here given a real committed artifact for the first time).
freeze-validloss-ci-9b-heterogeneous-generalization: ## GOAL §4 real-9B A/B on a HETEROGENEOUS (per-layer rank) stack, generalization regime
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci_9b $(FREEZE_9B_GEN_FLAGS) --architecture heterogeneous --json --output tests/fixtures/freeze_validloss_ci_9b_heterogeneous_generalization.json --run-log tests/fixtures/freeze_validloss_ci_9b_heterogeneous_generalization_runlog.json

# FULL-BACKPROP BASELINE arm of the real 9B §4 A/B — GOAL §4 line 247's OTHER
# success axis. The surrogate/direction/generalization deposits above are all
# freeze-vs-freeze (candidate output-first vs random surrogate vs input-side
# control); none compares against a NO-FREEZE full-backprop baseline. §4 line
# 247 requires "valid_loss degradation within tolerance of FULL backprop" — a
# condition the freeze-vs-freeze verdicts cannot speak to. This target adds a
# depth=0 baseline arm (every active-scope layer trained on full CE throughout;
# max_depth=0 plans zero freezes) alongside candidate+surrogate+control in the
# SAME generalization regime (48 train / 96 step / 2 epoch), so the deposit
# carries a single-session candidate-vs-baseline CI (the §4 line-247 axis) PLUS
# the existing candidate-vs-surrogate and direction verdicts for a complete
# 4-arm ranking. Verdict reading: SURPASSES (candidate < baseline — method beats
# full backprop) / TIES (within tolerance — quality maintained, the §4 target) /
# UNDERSHOOTS (candidate > baseline — freezing cost quality at this budget).
# Still reduced-budget (96 < max_steps=1500) and honest about it.
# Needs a torch+bnb+GPU interpreter: PYTHON_VENV=/path/to/torch-python make freeze-validloss-ci-9b-baseline
FREEZE_9B_BASELINE_FLAGS ?= --seq-len 1024 --total-steps 96 --warmup-steps 12 --depth 3 --spacing 10 --n-candidate 3 --n-surrogate 3 --n-control 3 --n-baseline 3 --train-examples 48 --valid-examples 32 --max-dataset-rows 600

freeze-validloss-ci-9b-baseline: ## GOAL §4 real-9B A/B + FULL-BACKPROP baseline (candidate vs no-freeze; §4 line-247 axis)
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci_9b $(FREEZE_9B_BASELINE_FLAGS) --json --output tests/fixtures/freeze_validloss_ci_9b_baseline.json

# The FULL-BUDGET real-9B §4 A/B — the path to the first citable_as_full_section4_verdict=True
# deposit. total_steps=1500 reaches the config's max_steps (so reduced_budget=False), and
# train_examples=600 keeps a 1500-step run at ~2.5 epochs = GENERALIZATION regime (the 4th
# honesty axis: a naive --total-steps 1500 --train-examples 48 would do ~31 epochs, memorize,
# and _classify_regime would correctly block the full-verdict gate). candidate+surrogate+baseline
# at 3 seeds/arm gives the headline A/B (surrogate), the §4-line-247 axis (baseline), all non-thin.
# HONEST CAVEAT: ~hours of 9B GPU (3 arms x 3 seeds x 1500 steps); run as a long background job
# on a free GPU. The private src.data quality filter is still absent on this mirror, so absolute
# loss levels differ from a filtered run (A/B internal validity is unaffected — same data both arms).
# Needs a torch+bnb+GPU interpreter: PYTHON_VENV=/path/to/torch-python make freeze-validloss-ci-9b-full
# --ledger turns this multi-hour run RESUMABLE: each completed arm streams to the
# JSONL ledger, so an interruption (GPU preemption by a concurrent run, OOM,
# session end) banks its progress and the next invocation skips the done arms
# and executes only the missing ones — the difference between the verdict
# landing incrementally across free-GPU windows vs. every interruption
# restarting all 9 arms from zero. The ledger lives under runs/ (gitignored).
# Background-citizen caveat: the script defers a held GPU with exit 75
# (EX_TEMPFAIL), but GNU make flattens EVERY non-zero recipe exit to make-exit
# 2 — so a poll-loop wrapper around `make` cannot see the 75 (it looks identical
# to the script's exit-2 CUDA-down and exit-3 torn-ledger). A background job
# that retries on tempfail must therefore bypass make and run the module
# directly (or sniff "Deferring — exit 75" in the output); the exit codes are
# only honored when the interpreter is invoked without a make wrapper.
FREEZE_9B_FULL_FLAGS ?= --seq-len 1024 --total-steps 1500 --warmup-steps 150 --depth 3 --spacing 450 --n-candidate 3 --n-surrogate 3 --n-baseline 3 --train-examples 600 --valid-examples 64 --max-dataset-rows 2000 --ledger runs/freeze_validloss_ci_9b_full_ledger.jsonl

freeze-validloss-ci-9b-full: ## GOAL §4 real-9B FULL-BUDGET A/B verdict (1500 steps; generalization regime; ~hours GPU)
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci_9b $(FREEZE_9B_FULL_FLAGS) --json --output tests/fixtures/freeze_validloss_ci_9b_full.json

# Self-retrying background launcher for freeze-validloss-ci-9b-full above. The
# direct target defers a held GPU with exit 75 (EX_TEMPFAIL) and banks completed
# arms in the --ledger, BUT the comment above documents the gap: GNU make
# flattens every non-zero recipe exit to make-exit 2, so a poll loop around
# `make freeze-validloss-ci-9b-full` cannot tell tempfail (75, retry) from
# CUDA-down (2, fatal) or torn-ledger (3, re-run) — they all look identical, so
# the documented "polite retryable background citizen" never actually had a way
# to be launched as one. This target closes that: scripts/launch_freeze_ci_9b_full
# bypasses make for the worker step (it subprocess-invokes the module directly so
# the exit codes survive), retries 75/3 with backoff, stops on 0/1/2, and is
# bounded (--max-attempts / --deadline-seconds) so a persistently-held GPU cannot
# spin forever. The worker flags are identical to the direct target — only the
# retry wrapper differs — so the SAME deposit (tests/fixtures/freeze_validloss_ci_9b_full.json)
# and SAME ledger (runs/freeze_validloss_ci_9b_full_ledger.jsonl) land either way.
# Run detached, e.g.:
#   nohup env PYTHON_VENV=/torch/venv/bin/python \
#     make freeze-validloss-ci-9b-full-bg >runs/full_bg.log 2>&1 &
# Override the retry policy via LAUNCH_FLAGS, e.g. LAUNCH_FLAGS="--max-attempts 200 --tempfail-sleep 90".
# Needs a torch+bnb+GPU interpreter: PYTHON_VENV=/path/to/torch-python make freeze-validloss-ci-9b-full-bg
LAUNCH_FLAGS ?=
freeze-validloss-ci-9b-full-bg: ## Self-retrying background launcher for the full-budget 9B verdict (polls a busy GPU, banks via --ledger)
	$(PYTHON_VENV) -m scripts.launch_freeze_ci_9b_full $(LAUNCH_FLAGS) -- $(FREEZE_9B_FULL_FLAGS) --json --output tests/fixtures/freeze_validloss_ci_9b_full.json

# The FULL-BUDGET × HETEROGENEOUS §4 leg — the ONE remaining open research task.
# The homogeneous full-budget verdict LANDED (TIES, commit 4b88ca8: the output-
# first ORDER effect does not survive full-budget generalization) and the
# heterogeneous REDUCED-budget verdict LANDED (SURPASSES, 96 steps). What no run
# has yet measured: does the heterogeneous SURPASSES survive the FULL budget
# (1500 steps, generalization regime) on an ASYMMETRIC per-layer-rank adapter?
# This target closes the launch-path gap so the open leg runs through the SAME
# robust bg-launcher infrastructure as the homogeneous full run the moment a GPU
# window opens. The harness already supports the combination — ``--architecture
# heterogeneous`` threads through arm_valid_loss_9b and the ledger fingerprint,
# the citation gate (_full_section4_verdict_gate) is architecture-agnostic, and
# scripts/launch_freeze_ci_9b_full is a generic pass-through — so NO harness
# change is needed; only the canonical flag composition + a DISTINCT deposit and
# ledger path (so the heterogeneous full verdict never clobbers the homogeneous
# full deposit/ledger). The arm shape mirrors the landed heterogeneous-
# generalization target (candidate + surrogate + input-side control = the
# direction-isolation A/B), bumped from 96 to 1500 steps; --total-steps 1500
# reaches the config max_steps so reduced_budget=False, and --train-examples 600
# keeps a 1500-step run at ~2.5 epochs = generalization regime (the 4th honesty
# axis a naive 1500/48 run would violate by memorizing). Verdict reading mirrors
# the homogeneous full leg: SURPASSES (order survives full budget) / TIES (order
# effect evaporates, as it did homogeneously) / UNDERSHOOTS (freeze cost quality).
# Needs a torch+bnb+GPU interpreter: PYTHON_VENV=/path/to/torch-python make freeze-validloss-ci-9b-full-heterogeneous
FREEZE_9B_FULL_HETEROGENEOUS_FLAGS ?= --seq-len 1024 --total-steps 1500 --warmup-steps 150 --depth 3 --spacing 450 --n-candidate 3 --n-surrogate 3 --n-control 3 --train-examples 600 --valid-examples 64 --max-dataset-rows 2000 --ledger runs/freeze_validloss_ci_9b_full_heterogeneous_ledger.jsonl

freeze-validloss-ci-9b-full-heterogeneous: ## GOAL §4 real-9B FULL-BUDGET A/B on a HETEROGENEOUS per-layer-rank stack (1500 steps; generalization; ~hours GPU)
	$(PYTHON_VENV) -m scripts.run_freeze_validloss_ci_9b $(FREEZE_9B_FULL_HETEROGENEOUS_FLAGS) --architecture heterogeneous --json --output tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json

# Self-retrying background launcher for the leg above — same wrapper as the
# homogeneous full run (subprocess-invokes the worker so exit codes survive,
# retries GPU-tempfail 75 / torn-ledger 3, stops on 0/1/2, bounded so a held GPU
# cannot spin forever, banks completed arms in --ledger). Identical worker flags
# to the direct target — only the retry wrapper differs — so the SAME deposit
# and ledger land either way. Run detached, e.g.:
#   nohup env PYTHON_VENV=/torch/venv/bin/python \
#     make freeze-validloss-ci-9b-full-heterogeneous-bg >runs/full_heterogeneous_bg.log 2>&1 &
# Needs a torch+bnb+GPU interpreter: PYTHON_VENV=/path/to/torch-python make freeze-validloss-ci-9b-full-heterogeneous-bg
freeze-validloss-ci-9b-full-heterogeneous-bg: ## Self-retrying background launcher for the full-budget heterogeneous 9B verdict (polls a busy GPU, banks via --ledger)
	$(PYTHON_VENV) -m scripts.launch_freeze_ci_9b_full $(LAUNCH_FLAGS) -- $(FREEZE_9B_FULL_HETEROGENEOUS_FLAGS) --architecture heterogeneous --json --output tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json


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

# Re-judge recorded valid_loss samples through the §4 judge with NO GPU and NO
# model — the Category-C step reduced to a concrete, executable command. Reads
# the JSON schema `run_freeze_validloss_ci --json` writes (and the same schema a
# future 9B target run deposits), so a committed recording is verifiable
# anywhere and a target-scale sample file drops straight in: the proxy_scale
# flag in the file upgrades the verdict's scale label with no code change. The
# default re-judges the committed proxy recording and asserts it still replays
# to its recorded TIES (override FREEZE_REPLAY_FLAGS for another file / verdict).
# Needs only numpy (NOT torch/GPU): PYTHON_VENV=/path/to/numpy-python make freeze-replay
FREEZE_REPLAY_FLAGS ?= tests/fixtures/freeze_validloss_generalize_proxy.json --expected TIES

freeze-replay: ## Re-judge recorded valid_loss samples through the §4 judge (no GPU); asserts the recording still replays to its verdict
	$(PYTHON_VENV) -m scripts.replay_freeze_validloss_ci $(FREEZE_REPLAY_FLAGS)

# Turnkey for recipe TASK-0152 Tier-1 step 3: form a §4 verdict-gate deposit from
# real upstream run_metrics.jsonl (candidate = output_first progressive-freeze vs
# surrogate = full-backprop baseline, multi-seed) so best_valid_loss is read
# straight from the artifact — not hand-transcribed (a P0 reproducibility hazard).
# Default prints --help; override FREEZE_FORM_DEPOSIT_FLAGS with the real
# --candidate/--surrogate/--model/--device/--output, then `make freeze-replay`
# the emitted file to judge it GPU-free.
FREEZE_FORM_DEPOSIT_FLAGS ?= --help

freeze-form-deposit: ## Form a §4 verdict-gate deposit from real upstream run_metrics.jsonl (TASK-0152 Tier-1 step 3; override FREEZE_FORM_DEPOSIT_FLAGS)
	$(PYTHON_VENV) -m scripts.form_freeze_validloss_deposit $(FREEZE_FORM_DEPOSIT_FLAGS)

# Re-derive the recorded order-sensitivity decomposition (Var(order)/Var(seed))
# from stored per-arm samples with NO GPU, NO model, and NO torch — the other
# half of the proxy evidence, given the same GPU-free-replay treatment as
# freeze-replay. The verdict replay pins the recorded TIES; this pins the
# recorded ratio=0.000 — the result that converts "target-scale is assumed
# necessary" into "proven necessary". Reads the JSON schema
# `run_freeze_order_sensitivity --json` writes (and the same schema a future 9B
# target run deposits), so a committed recording is verifiable anywhere. The
# default re-judges the committed proxy recording and asserts it still replays
# to its recorded not_resolvable (override FREEZE_ORDER_REPLAY_FLAGS for another
# file / outcome). Needs only the stdlib (NOT torch/GPU/numpy).
FREEZE_ORDER_REPLAY_FLAGS ?= tests/fixtures/freeze_order_sensitivity_proxy.json --expected not_resolvable

freeze-order-sensitivity-replay: ## Re-derive the recorded order-sensitivity decomposition (no GPU/torch); asserts the recording still replays to its outcome
	$(PYTHON_VENV) -m scripts.replay_freeze_order_sensitivity $(FREEZE_ORDER_REPLAY_FLAGS)

