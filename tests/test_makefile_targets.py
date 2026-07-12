"""Verify Makefile smoke, ablation, bench-optimizer, and experiment config target wiring."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.training.config_schema import (BaselineConfig, TGLoRAConfig,
                                        load_and_validate_config)

REPO_ROOT = Path(__file__).resolve().parent.parent
MAKEFILE = REPO_ROOT / "Makefile"


@pytest.fixture()
def makefile_content() -> str:
    return MAKEFILE.read_text()


@pytest.fixture()
def makefile_lines() -> list[str]:
    return MAKEFILE.read_text().splitlines()


def _find_target_body(lines: list[str], target: str) -> list[str]:
    """Return the recipe lines for *target*, handling ``\\`` continuations."""
    body: list[str] = []
    found = False
    continuing = False
    for line in lines:
        if found:
            if not continuing and line and line[0] not in ("\t", " ") and line.strip():
                break
            body.append(line)
            continuing = line.rstrip().endswith("\\")
        if line.startswith(f"{target}:"):
            found = True
    return body


# ── Target existence ────────────────────────────────────────────────────────


class TestTargetDefined:
    def test_smoke_tg_target_defined(self, makefile_content: str) -> None:
        assert "smoke-tg:" in makefile_content

    def test_smoke_bl_target_defined(self, makefile_content: str) -> None:
        assert "smoke-bl:" in makefile_content

    def test_bench_optimizer_target_defined(self, makefile_content: str) -> None:
        assert "bench-optimizer:" in makefile_content

    def test_ablation_target_defined(self, makefile_content: str) -> None:
        assert "ablation:" in makefile_content

    def test_paper_memory_target_defined(self, makefile_content: str) -> None:
        assert "paper-memory:" in makefile_content

    def test_paper_memory_one_shot_target_defined(self, makefile_content: str) -> None:
        assert "paper-memory-one-shot:" in makefile_content

    def test_paper_memory_compare_modes_target_defined(self, makefile_content: str) -> None:
        assert "paper-memory-compare-modes:" in makefile_content

    def test_paper_memory_all_modes_target_defined(self, makefile_content: str) -> None:
        assert "paper-memory-all-modes:" in makefile_content

    def test_precompute_prefix_cache_target_defined(self, makefile_content: str) -> None:
        assert "precompute-prefix-cache:" in makefile_content

    def test_bench_prefix_cache_one_shot_target_defined(self, makefile_content: str) -> None:
        assert "bench-prefix-cache-one-shot:" in makefile_content

    def test_analyze_prefix_break_even_target_defined(self, makefile_content: str) -> None:
        assert "analyze-prefix-break-even:" in makefile_content


# ── smoke-tg wiring ────────────────────────────────────────────────────────


class TestSmokeTg:
    def test_copies_config(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "smoke-tg")
        body_text = "\n".join(body)
        assert "configs/9b_tg_lora.yaml" in body_text
        assert "runs/smoke_tg/config.yaml" in body_text

    def test_invokes_training_module(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "smoke-tg")
        body_text = "\n".join(body)
        assert "src.training.train_tg_lora" in body_text

    def test_run_dir_created(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "smoke-tg")
        body_text = "\n".join(body)
        assert "runs/smoke_tg" in body_text

    def test_config_source_exists(self) -> None:
        assert (REPO_ROOT / "configs/9b_tg_lora.yaml").is_file()


# ── smoke-bl wiring ────────────────────────────────────────────────────────


class TestSmokeBl:
    def test_copies_config(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "smoke-bl")
        body_text = "\n".join(body)
        assert "configs/9b_baseline.yaml" in body_text
        assert "runs/smoke_bl/config.yaml" in body_text

    def test_invokes_training_module(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "smoke-bl")
        body_text = "\n".join(body)
        assert "src.training.train_baseline_qlora" in body_text

    def test_run_dir_created(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "smoke-bl")
        body_text = "\n".join(body)
        assert "runs/smoke_bl" in body_text

    def test_config_source_exists(self) -> None:
        assert (REPO_ROOT / "configs/9b_baseline.yaml").is_file()


# ── bench-optimizer wiring ─────────────────────────────────────────────────


class TestBenchOptimizer:
    def test_calls_script(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "bench-optimizer")
        body_text = "\n".join(body)
        assert "scripts/benchmark_optimizer_lifecycle.py" in body_text

    def test_script_exists(self) -> None:
        assert (REPO_ROOT / "scripts/benchmark_optimizer_lifecycle.py").is_file()

    def test_script_outputs_json(self) -> None:
        """Verify the script contains json.dumps output (prints JSON to stdout)."""
        script = (REPO_ROOT / "scripts/benchmark_optimizer_lifecycle.py").read_text()
        assert "json.dumps" in script


# ── ablation wiring ────────────────────────────────────────────────────────


class TestAblation:
    def test_calls_script(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "ablation")
        body_text = "\n".join(body)
        assert "scripts/run_ablation_suite.sh" in body_text

    def test_script_exists(self) -> None:
        assert (REPO_ROOT / "scripts/run_ablation_suite.sh").is_file()

    def test_script_is_executable(self) -> None:
        script = REPO_ROOT / "scripts/run_ablation_suite.sh"
        assert script.stat().st_mode & 0o111, "run_ablation_suite.sh must be executable"

    def test_script_writes_metrics(self) -> None:
        """Verify the ablation script references run_metrics.jsonl output."""
        script = (REPO_ROOT / "scripts/run_ablation_suite.sh").read_text()
        assert "run_metrics.jsonl" in script


class TestPaperMemory:
    def test_calls_script(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "paper-memory")
        body_text = "\n".join(body)
        assert "scripts/run_paper_memory_suite.sh" in body_text

    def test_script_exists(self) -> None:
        assert (REPO_ROOT / "scripts/run_paper_memory_suite.sh").is_file()

    def test_references_frozen_prefix_cache_config(self) -> None:
        script = (REPO_ROOT / "scripts/run_paper_memory_suite.sh").read_text()
        assert "configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml" in script

    def test_references_suffix_only_baseline(self) -> None:
        script = (REPO_ROOT / "scripts/run_paper_memory_suite.sh").read_text()
        assert "configs/9b_baseline_suffix_only_last25.yaml" in script

    def test_forwards_cuda_visible_devices_when_set(self) -> None:
        script = (REPO_ROOT / "scripts/run_paper_memory_suite.sh").read_text()
        assert "--cuda-visible-devices" in script
        assert "CUDA_VISIBLE_DEVICES_VALUE" in script

    def test_paper_memory_one_shot_uses_one_shot_config(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "paper-memory-one-shot")
        body_text = "\n".join(body)
        assert "configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml" in body_text

    def test_paper_memory_one_shot_uses_separate_cache_base(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "paper-memory-one-shot")
        body_text = "\n".join(body)
        assert ".cache/prefix_feature_cache_paper_suite_one_shot" in body_text

    def test_paper_memory_all_modes_runs_both_variants(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "paper-memory-all-modes")
        body_text = "\n".join(body)
        assert "$(MAKE) paper-memory" in body_text
        assert "$(MAKE) paper-memory-one-shot" in body_text
        assert "$(MAKE) paper-memory-compare-modes" in body_text

    def test_paper_memory_all_modes_uses_aggregate_summaries(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "paper-memory-all-modes")
        body_text = "\n".join(body)
        assert "reuse/aggregate_summary.json" in body_text
        assert "one_shot/aggregate_summary.json" in body_text


class TestParallelPrecompute:
    def test_calls_script(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "precompute-prefix-cache")
        body_text = "\n".join(body)
        assert "scripts/precompute_prefix_cache_parallel.py" in body_text

    def test_script_exists(self) -> None:
        assert (REPO_ROOT / "scripts/precompute_prefix_cache_parallel.py").is_file()

    def test_references_frozen_prefix_cache_config(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "precompute-prefix-cache")
        body_text = "\n".join(body)
        assert "configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml" in body_text


class TestBreakEvenAnalysis:
    def test_calls_script(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "analyze-prefix-break-even")
        body_text = "\n".join(body)
        assert "scripts/analyze_prefix_cache_break_even.py" in body_text

    def test_script_exists(self) -> None:
        assert (REPO_ROOT / "scripts/analyze_prefix_cache_break_even.py").is_file()


class TestBenchPrefixCacheTargets:
    def test_bench_prefix_cache_accepts_tg_config_override(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "bench-prefix-cache")
        body_text = "\n".join(body)
        assert "--tg-config $(or $(TG_CONFIG),configs/9b_tg_lora_prefix_feature_cache_experimental.yaml)" in body_text

    def test_bench_prefix_cache_accepts_cuda_visible_devices_override(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "bench-prefix-cache")
        body_text = "\n".join(body)
        assert "$(if $(CUDA_VISIBLE_DEVICES),--cuda-visible-devices $(CUDA_VISIBLE_DEVICES))" in body_text

    def test_bench_prefix_cache_one_shot_references_one_shot_config(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "bench-prefix-cache-one-shot")
        body_text = "\n".join(body)
        assert "configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml" in body_text


class TestCompareTargets:
    def test_compare_prefix_exports_cuda_visible_devices(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "compare-prefix")
        body_text = "\n".join(body)
        assert "CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),)" in body_text

    def test_paper_memory_exports_cuda_visible_devices(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "paper-memory")
        body_text = "\n".join(body)
        assert "CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),)" in body_text


# ── smoke target produces run_metrics.jsonl (wiring check) ─────────────────


class TestSmokeMetricsWiring:
    """Verify the training modules invoked by smoke targets write run_metrics.jsonl."""

    def test_tg_lora_training_writes_metrics(self) -> None:
        src = (REPO_ROOT / "src/training/train_tg_lora.py").read_text()
        assert "run_metrics" in src.lower() or "RunMetrics" in src

    def test_baseline_training_writes_metrics(self) -> None:
        src = (REPO_ROOT / "src/training/train_baseline_qlora.py").read_text()
        assert "run_metrics" in src.lower() or "RunMetrics" in src

    def test_run_metrics_module_writes_jsonl(self) -> None:
        src = (REPO_ROOT / "src/utils/run_metrics.py").read_text()
        assert "run_metrics.jsonl" in src


# ── Experiment config schema validation (TASK-0074) ──────────────────────────


class TestConfigSchemaValidation:
    """Verify experimental YAML configs pass Pydantic schema validation."""

    def test_suffix_only_baseline_config_validates(self) -> None:
        config = load_and_validate_config(
            REPO_ROOT / "configs/9b_baseline_suffix_only_last25.yaml"
        )
        assert isinstance(config, BaselineConfig)
        assert config.experiment.name == "qlora_9b_baseline_suffix_only_last25"

    def test_prefix_feature_cache_config_validates(self) -> None:
        config = load_and_validate_config(
            REPO_ROOT
            / "configs/9b_tg_lora_prefix_feature_cache_experimental.yaml"
        )
        assert isinstance(config, TGLoRAConfig)
        assert (
            config.experiment.name
            == "qwen35_9b_tg_lora_prefix_feature_cache_experimental"
        )
        assert config.training.prefix_feature_cache_experimental is True

    def test_optimizer_reuse_config_validates(self) -> None:
        config = load_and_validate_config(
            REPO_ROOT
            / "configs/9b_tg_lora_optimizer_reuse_experimental.yaml"
        )
        assert isinstance(config, TGLoRAConfig)
        assert (
            config.training.optimizer_lifecycle
            == "reuse_state_reset_experimental"
        )

    def test_paper_poc_config_validates(self) -> None:
        config = load_and_validate_config(
            REPO_ROOT / "configs/9b_tg_lora_paper_poc.yaml"
        )
        assert isinstance(config, TGLoRAConfig)
        assert config.experiment.name == "tg_lora_9b_paper_poc"

    def test_prefix_feature_cache_paper_poc_config_validates(self) -> None:
        config = load_and_validate_config(
            REPO_ROOT / "configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml"
        )
        assert isinstance(config, TGLoRAConfig)
        assert (
            config.experiment.name
            == "qwen35_9b_tg_lora_prefix_feature_cache_paper_poc"
        )
        assert config.training.prefix_feature_cache_experimental is True
        assert config.training.prefix_feature_cache_train is True
        assert config.training.prefix_feature_cache_mode == "reuse"
        assert config.training.prefix_feature_cache_share_across_seeds is True
        assert config.training.prefix_feature_cache_offload_prefix_to_cpu is True
        assert config.tg_lora.enable_random_walk is False
        assert config.tg_lora.enable_convergence_adaptation is False

    def test_prefix_feature_cache_one_shot_poc_config_validates(self) -> None:
        config = load_and_validate_config(
            REPO_ROOT / "configs/9b_tg_lora_prefix_feature_cache_one_shot_poc.yaml"
        )
        assert isinstance(config, TGLoRAConfig)
        assert (
            config.experiment.name
            == "qwen35_9b_tg_lora_prefix_feature_cache_one_shot_poc"
        )
        assert config.training.prefix_feature_cache_experimental is True
        assert config.training.prefix_feature_cache_mode == "one_shot"
        assert config.training.prefix_feature_cache_share_across_seeds is False
        assert config.training.prefix_feature_cache_offload_prefix_to_cpu is True
        assert config.training.prefix_feature_cache_num_workers == 0



# ── Experiment config Makefile wiring (TASK-0074) ────────────────────────────


class TestOptreuseTarget:
    """Verify train-tg-lora-optreuse target wiring."""

    def test_references_correct_yaml(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "train-tg-lora-optreuse")
        body_text = "\n".join(body)
        assert "configs/9b_tg_lora_optimizer_reuse_experimental.yaml" in body_text

    def test_yaml_file_exists(self) -> None:
        assert (
            REPO_ROOT / "configs/9b_tg_lora_optimizer_reuse_experimental.yaml"
        ).is_file()

    def test_invokes_correct_module(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "train-tg-lora-optreuse")
        body_text = "\n".join(body)
        assert "src.training.train_tg_lora" in body_text

    def test_module_exists(self) -> None:
        assert (REPO_ROOT / "src/training/train_tg_lora.py").is_file()


class TestPrefixTarget:
    """Verify train-tg-lora-prefix target wiring."""

    def test_references_correct_yaml(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "train-tg-lora-prefix")
        body_text = "\n".join(body)
        assert (
            "configs/9b_tg_lora_prefix_feature_cache_experimental.yaml"
            in body_text
        )

    def test_yaml_file_exists(self) -> None:
        assert (
            REPO_ROOT
            / "configs/9b_tg_lora_prefix_feature_cache_experimental.yaml"
        ).is_file()

    def test_invokes_correct_module(self, makefile_lines: list[str]) -> None:
        body = _find_target_body(makefile_lines, "train-tg-lora-prefix")
        body_text = "\n".join(body)
        assert "src.training.train_tg_lora" in body_text

    def test_module_exists(self) -> None:
        assert (REPO_ROOT / "src/training/train_tg_lora.py").is_file()


# ── PYTHON_VENV override resolution (launch-path exit-127 regression) ────────


class TestPythonVenvOverride:
    """``PYTHON_VENV=/path make <gpu-target>`` — the env-var override form
    prescribed by all 8 "Needs a torch+bnb+GPU interpreter" Makefile comments
    and the verdict-run launcher — must actually override the interpreter.

    Regression: ``PYTHON_VENV := $(VENV)/bin/python`` (a makefile ``:=``)
    silently clobbered the env-var override back to ``.venv/bin/python``, which
    exits 127 on any checkout with no ``.venv`` (worktrees, fresh clones, CI).
    The first launch of ``freeze-validloss-ci-9b-full`` hit exactly this — the
    documented GPU launch form never fired. Same exit-127-on-fresh-checkout
    family as ``TestPaperMemoryDryRun`` (TASK-0154). The ``?=`` fix lets an
    env/command-line ``PYTHON_VENV`` win while preserving the default.
    """

    _TARGET = "freeze-validloss-ci-9b-full"
    _OVERRIDE = "/opt/torch-bnb-venv/bin/python"

    def test_assignment_is_conditional_not_immediate(
        self, makefile_lines: list[str]
    ) -> None:
        """``?=`` is load-bearing: ``:=`` would clobber the env-var override."""
        line = next(
            (ln for ln in makefile_lines if ln.startswith("PYTHON_VENV")), None
        )
        assert line is not None, "PYTHON_VENV assignment missing from Makefile"
        assert line.startswith("PYTHON_VENV ?="), (
            "PYTHON_VENV must use ?= (not :=) so 'PYTHON_VENV=/path make ...' "
            f"overrides the interpreter; got: {line!r}"
        )

    def test_env_var_override_resolves(self) -> None:
        """``PYTHON_VENV=/path make -n`` must invoke /path (was silently ignored)."""
        if shutil.which("make") is None:
            pytest.skip("make unavailable")
        result = subprocess.run(
            ["make", "-n", self._TARGET],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHON_VENV": self._OVERRIDE},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert self._OVERRIDE in result.stdout
        # The default must NOT leak through when the override is set.
        assert ".venv/bin/python" not in result.stdout

    def test_command_line_override_resolves(self) -> None:
        """``make PYTHON_VENV=/path -n`` must invoke /path."""
        if shutil.which("make") is None:
            pytest.skip("make unavailable")
        env = {k: v for k, v in os.environ.items() if k != "PYTHON_VENV"}
        result = subprocess.run(
            ["make", "-n", f"PYTHON_VENV={self._OVERRIDE}", self._TARGET],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert self._OVERRIDE in result.stdout

    def test_default_interpreter_unchanged(self) -> None:
        """With no override the interpreter stays ``.venv/bin/python``."""
        if shutil.which("make") is None:
            pytest.skip("make unavailable")
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("PYTHON_VENV", "VENV", "MAKEFLAGS")
        }
        result = subprocess.run(
            ["make", "-n", self._TARGET],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert ".venv/bin/python" in result.stdout


# ── GPU-free dry-run validation (TASK-0154) ─────────────────────────────────


class TestPaperMemoryDryRun:
    """``make paper-memory-dry-run`` must validate the suffix-only verdict
    suite's config assembly without a GPU *or* a materialized ``.venv``.

    Regression for the ``VENV_PYTHON`` interpreter fallback in
    ``scripts/run_paper_memory_suite.sh``: the suite's OmegaConf seed-patch step
    resolved to ``.venv/bin/python``, so on any fresh clone (CI included) where
    that path is absent the dry-run exited 127 *before* reporting any validation
    — the very tool meant to prove the verdict path assembles GPU-free was itself
    broken. The fix falls back to a discoverable ``python3``.
    """

    def test_dry_run_validates_when_venv_python_absent(self, tmp_path: Path) -> None:
        pytest.importorskip("omegaconf")
        if shutil.which("bash") is None or shutil.which("python3") is None:
            pytest.skip("bash/python3 unavailable")
        # Force the fallback branch: an interpreter path that does not exist, so
        # the script must discover ``python3`` rather than exit 127.
        absent_python = tmp_path / "absent_venv" / "bin" / "python"
        env = {
            **os.environ,
            "DRY_RUN": "true",
            "SEEDS": "42",
            "OUTPUT_BASE": str(tmp_path / "dryrun"),
            "BASELINE_CONFIG": "configs/9b_baseline_suffix_only_last25.yaml",
            "TG_CONFIG": "configs/9b_tg_lora_prefix_feature_cache_paper_poc.yaml",
            "VENV_PYTHON": str(absent_python),
        }
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts" / "run_paper_memory_suite.sh")],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert "[OK]" in result.stdout
        assert "DRY RUN complete" in result.stdout
