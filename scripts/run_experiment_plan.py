#!/usr/bin/env python
"""Run a paper experiment plan end-to-end.

Loads an experiment plan config, validates the sub-configs, executes the benchmark
suite (paper-memory or frontier-sweep), performs downstream evaluation, ingests the
evidence, performs quality checks, and compiles the paper PDF.
"""
import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Add repository root to python path to allow imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

def load_yaml(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to load YAML at {path}: {e}")
        sys.exit(1)

def save_yaml(data, path: Path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False)
    except Exception as e:
        print(f"ERROR: Failed to save YAML to {path}: {e}")
        sys.exit(1)

def validate_and_patch_config(config_path: Path, experiment_id: str):
    """Validate that the config has paper_experiment: true and correct paper_experiment_id."""
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
        
    config = load_yaml(config_path)
    if 'experiment' not in config:
        config['experiment'] = {}
        
    exp = config['experiment']
    modified = False
    
    if not exp.get('paper_experiment', False):
        print(f"Warning: Config {config_path} did not have 'paper_experiment: true'. Setting it.")
        exp['paper_experiment'] = True
        modified = True
        
    if exp.get('paper_experiment_id') != experiment_id:
        print(f"Warning: Config {config_path} had paper_experiment_id '{exp.get('paper_experiment_id')}' which differs from plan '{experiment_id}'. Updating it.")
        exp['paper_experiment_id'] = experiment_id
        modified = True
        
    if modified:
        save_yaml(config, config_path)
        print(f"Successfully patched config {config_path}")

def run_command(cmd, env=None, cwd=None):
    print(f"\nRunning command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, env=env, cwd=cwd, shell=isinstance(cmd, str))
    if res.returncode != 0:
        print(f"ERROR: Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    return res

def main():
    parser = argparse.ArgumentParser(description="Execute a paper experiment plan end-to-end")
    parser.add_argument("--config", type=str, default="configs/paper_experiment_plan.yaml", help="Path to experiment plan YAML")
    parser.add_argument("--skip-eval", action="store_true", help="Skip running the downstream evaluations")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip compiling the paper PDF")
    args = parser.parse_args()

    plan_path = Path(args.config).resolve()
    if not plan_path.exists():
        print(f"ERROR: Experiment plan config not found at {plan_path}")
        sys.exit(1)

    print(f"Loading experiment plan from {plan_path}...")
    plan_data = load_yaml(plan_path)
    plan = plan_data.get('experiment_plan')
    if not plan:
        print("ERROR: YAML must contain 'experiment_plan' root element")
        sys.exit(1)

    experiment_id = plan.get('experiment_id')
    plan_type = plan.get('type')
    seeds = plan.get('seeds', [42, 43, 44])
    baseline_cfg = repo_root / plan.get('baseline_config')
    tg_cfg = repo_root / plan.get('tg_config')
    cache_base = plan.get('cache_base', '.cache/prefix_feature_cache_paper_suite')
    
    params = plan.get('parameters', {})
    target_bp = params.get('target_bp', 240)
    max_seq_len = params.get('max_seq_len', 1024)
    quick_eval_examples = params.get('quick_eval_examples', 32)
    eval_points = params.get('eval_points', 3)
    mlflow_enabled = str(params.get('mlflow_enabled', False)).lower()

    # Define unique timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = repo_root / "runs" / f"{experiment_id}_{timestamp}"
    
    print("\n==================================================")
    print(f"  Executing Plan: {experiment_id} ({plan_type})")
    print(f"  Output Base: {output_base}")
    print(f"  Seeds: {seeds}")
    print("==================================================")

    # 1. Validate & Patch baseline and tg configs to align experiment IDs
    validate_and_patch_config(baseline_cfg, experiment_id)
    validate_and_patch_config(tg_cfg, experiment_id)

    # 2. Setup run environment variables
    env = os.environ.copy()
    env["TARGET_BP"] = str(target_bp)
    env["MAX_SEQ_LEN"] = str(max_seq_len)
    env["QUICK_EVAL_EXAMPLES"] = str(quick_eval_examples)
    env["EVAL_POINTS"] = str(eval_points)
    env["SEEDS"] = " ".join(str(s) for s in seeds)
    env["OUTPUT_BASE"] = str(output_base)
    env["BASELINE_CONFIG"] = str(baseline_cfg)
    env["TG_CONFIG"] = str(tg_cfg)
    env["CACHE_BASE"] = str(cache_base)
    env["MLFLOW_ENABLED"] = mlflow_enabled
    env["VENV_PYTHON"] = str(repo_root / ".venv/bin/python")

    # 3. Execute core experiment suite
    if plan_type == "paper-memory":
        script_path = repo_root / "scripts/run_paper_memory_suite.sh"
        run_command([str(script_path)], env=env, cwd=str(repo_root))
    elif plan_type == "frontier-sweep":
        script_path = repo_root / "scripts/run_frontier_sweep.sh"
        run_command([str(script_path)], env=env, cwd=str(repo_root))
    else:
        print(f"ERROR: Unknown plan type: {plan_type}")
        sys.exit(1)

    # 4. Run Downstream Evaluation
    eval_cfg = plan.get('evaluation', {})
    run_eval = eval_cfg.get('run_external_eval', False) and not args.skip_eval
    
    if run_eval and plan_type == "paper-memory":
        print("\n--- Running Downstream Evaluation on all seeds ---")
        summary_path = output_base / "aggregate_summary.json"
        eval_script = repo_root / "scripts/run_all_seeds_eval.py"
        run_command([str(repo_root / ".venv/bin/python"), str(eval_script), "--summary-path", str(summary_path)], cwd=str(repo_root))
    elif run_eval:
        print(f"\n[INFO] Downstream evaluation not automated for type '{plan_type}' (only supported for paper-memory)")

    # 5. Ingest Evidence
    print("\n--- Ingesting Paper Evidence ---")
    ingest_script = repo_root / "scripts/ingest_paper_evidence.py"
    run_command([str(repo_root / ".venv/bin/python"), str(ingest_script)], cwd=str(repo_root))

    # 6. Validate Evidence (check-evidence)
    print("\n--- Validating Ingested Evidence ---")
    run_command(["make", "check-evidence"], cwd=str(repo_root))

    # 7. Compile paper PDF using helix_scholar if configured
    scholar_cfg = plan.get('scholar', {})
    compile_pdf = scholar_cfg.get('compile_pdf', False) and not args.skip_pdf
    scholar_project = scholar_cfg.get('project')
    scholar_dir = Path("/home/jinno/helix_scholar")
    
    if compile_pdf and scholar_project and scholar_dir.exists():
        print(f"\n--- Compiling Paper PDF for project '{scholar_project}' ---")
        # Run helix package --project <project>
        helix_bin = repo_root / ".venv/bin/helix"
        run_command([str(helix_bin), "package", "--project", scholar_project], cwd=str(scholar_dir))
    elif compile_pdf:
        print("\n[WARNING] Could not compile PDF: scholar project not configured or helix_scholar path missing")

    print("\n==================================================")
    print(f"  Plan execution COMPLETE: {experiment_id}")
    print("  All evidence has been processed & verified.")
    print("==================================================")

if __name__ == "__main__":
    main()
