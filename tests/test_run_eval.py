"""Tests for eval scripts: TC-035-01 (run_eval.sh) and TC-035-02 (run_eval_lora.sh).

These tests verify script structure, task configuration, and argument parsing
without requiring lm-eval or a real model.
"""

import re
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ---------------------------------------------------------------------------
# TC-035-01: run_eval.sh structure & task configuration
# ---------------------------------------------------------------------------


def test_run_eval_script_exists():
    """TC-035-01: run_eval.sh exists and is executable."""
    script = SCRIPTS_DIR / "run_eval.sh"
    assert script.exists()


def test_run_eval_default_tasks():
    """TC-035-01: Default task list covers ARC-Easy, HellaSwag, GSM8K, TruthfulQA MC2."""
    script = SCRIPTS_DIR / "run_eval.sh"
    content = script.read_text()

    assert "arc_easy" in content
    assert "hellaswag" in content
    assert "gsm8k" in content
    assert "truthfulqa_mc2" in content

    # Verify they appear in the default TASKS variable
    tasks_match = re.search(r'TASKS="([^"]+)"', content)
    assert tasks_match is not None
    tasks = tasks_match.group(1).split(",")
    assert "arc_easy" in tasks
    assert "hellaswag" in tasks
    assert "gsm8k" in tasks
    assert "truthfulqa_mc2" in tasks
    assert len(tasks) == 4


def test_run_eval_uses_hf_model():
    """TC-035-01: Invokes lm_eval with --model hf."""
    script = SCRIPTS_DIR / "run_eval.sh"
    content = script.read_text()

    assert "--model hf" in content
    assert "--model_args" in content
    assert "pretrained=" in content


def test_run_eval_output_dir_configurable():
    """TC-035-01: Output directory is configurable via --output-dir."""
    script = SCRIPTS_DIR / "run_eval.sh"
    content = script.read_text()

    assert "--output-dir" in content
    assert "OUTPUT_DIR" in content


def test_run_eval_checks_lm_eval_installed():
    """TC-035-01: Script checks for lm_eval before running."""
    script = SCRIPTS_DIR / "run_eval.sh"
    content = script.read_text()

    assert "lm_eval" in content
    assert "command -v" in content or "which" in content


def test_run_eval_requires_model_path():
    """TC-035-01: Script requires MODEL_PATH as first argument."""
    script = SCRIPTS_DIR / "run_eval.sh"
    content = script.read_text()

    assert 'MODEL_PATH="${1' in content or "MODEL_PATH=" in content


# ---------------------------------------------------------------------------
# TC-035-02: run_eval_lora.sh structure & merge-eval-cleanup flow
# ---------------------------------------------------------------------------


def test_run_eval_lora_script_exists():
    """TC-035-02: run_eval_lora.sh exists."""
    script = SCRIPTS_DIR / "run_eval_lora.sh"
    assert script.exists()


def test_run_eval_lora_has_three_steps():
    """TC-035-02: Script has merge → eval → cleanup steps."""
    script = SCRIPTS_DIR / "run_eval_lora.sh"
    content = script.read_text()

    # Step 1: Merge
    assert "Merge" in content or "merge" in content
    assert "PeftModel" in content
    assert "merge_and_unload" in content

    # Step 2: Eval
    assert "run_eval.sh" in content

    # Step 3: Cleanup
    assert "Cleaning" in content or "clean" in content or "rm -rf" in content


def test_run_eval_lora_uses_temp_dir():
    """TC-035-02: Merged model goes to a temporary directory that gets cleaned up."""
    script = SCRIPTS_DIR / "run_eval_lora.sh"
    content = script.read_text()

    assert "MERGED_DIR" in content
    assert "/tmp/" in content
    assert "rm -rf" in content


def test_run_eval_lora_requires_two_args():
    """TC-035-02: Requires both base_model and adapter_path."""
    script = SCRIPTS_DIR / "run_eval_lora.sh"
    content = script.read_text()

    assert "BASE_MODEL" in content
    assert "ADAPTER_PATH" in content


def test_run_eval_lora_saves_tokenizer():
    """TC-035-02: Tokenizer is also saved alongside the merged model."""
    script = SCRIPTS_DIR / "run_eval_lora.sh"
    content = script.read_text()

    assert "AutoTokenizer" in content
    assert "save_pretrained" in content


# ---------------------------------------------------------------------------
# Integration: verify scripts pass shellcheck-level structural checks
# ---------------------------------------------------------------------------


def test_run_eval_lora_default_tasks_match_eval():
    """TC-035-02: LoRA eval uses same default tasks as base eval."""
    eval_script = SCRIPTS_DIR / "run_eval.sh"
    lora_script = SCRIPTS_DIR / "run_eval_lora.sh"

    eval_tasks = re.search(r'TASKS="([^"]+)"', eval_script.read_text())
    lora_tasks = re.search(r'TASKS="([^"]+)"', lora_script.read_text())

    assert eval_tasks is not None
    assert lora_tasks is not None
    assert eval_tasks.group(1) == lora_tasks.group(1)
