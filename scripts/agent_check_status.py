#!/usr/bin/env python3
import json
import os
from pathlib import Path

def count_lines(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def check_datasets():
    print("=== [1/3] Dataset Integrity Check ===")
    data_dir = Path("data")
    expected = {
        "train.jsonl": 4500,
        "valid_quick.jsonl": 450,
        "test.jsonl": 450,
    }
    
    all_ok = True
    for filename, min_lines in expected.items():
        path = data_dir / filename
        if not path.exists():
            print(f"[-] {filename}: MISSING")
            all_ok = False
        else:
            lines = count_lines(path)
            if lines < min_lines:
                print(f"[-] {filename}: Present but short ({lines}/{min_lines} lines)")
                all_ok = False
            else:
                print(f"[+] {filename}: OK ({lines} lines)")
                
    # Check MLX data link
    mlx_dir = Path("data_mlx")
    if mlx_dir.exists():
        train_link = mlx_dir / "train.jsonl"
        valid_link = mlx_dir / "valid.jsonl"
        if not train_link.exists() or not valid_link.exists():
            print("[-] MLX symlinks: INCOMPLETE or MISSING")
            all_ok = False
        else:
            print("[+] MLX symlinks: OK")
    else:
        print("[ ] MLX data directory not found (Track B not configured on this machine)")
        
    return all_ok

def check_experiment_runs():
    print("\n=== [2/3] Experiment Runs & Aggregate Summaries ===")
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("[-] runs/ directory does not exist.")
        return None
        
    # Look for aggregate summaries or paper-memory suites
    suites = list(runs_dir.glob("paper_memory_suite_*"))
    one_shot_suites = list(runs_dir.glob("paper_memory_one_shot_suite_*"))
    all_suites = sorted(suites + one_shot_suites, key=os.path.getmtime, reverse=True)
    
    if not all_suites:
        print("[-] No paper-memory suites found in runs/")
        return None
        
    print(f"Found {len(all_suites)} paper-memory suite runs. Most recent:")
    most_recent = all_suites[0]
    print(f"  Path: {most_recent}")
    
    # Check if aggregate_summary.json exists in the most recent run or outputs
    summary_path = most_recent / "aggregate_summary.json"
    if not summary_path.exists():
        # Check other runs
        print("  [-] aggregate_summary.json not found in the most recent run.")
        # Try to find any aggregate_summary.json
        all_summaries = sorted(list(runs_dir.glob("**/aggregate_summary.json")), key=os.path.getmtime, reverse=True)
        if all_summaries:
            summary_path = all_summaries[0]
            print(f"  [+] Found an aggregate summary in a different run: {summary_path}")
        else:
            summary_path = None
            
    if summary_path and summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
            print(f"[+] Loaded {summary_path.name}")
            return data
        except Exception as e:
            print(f"[-] Failed to parse summary json: {e}")
            
    return None

def evaluate_and_suggest(data_ok, summary_data):
    print("\n=== [3/3] Milestone & Next Step Evaluation ===")
    
    if not data_ok:
        print("Recommendation:")
        print("  -> Run data preparation to set up the 5K Dolly dataset split.")
        print("  Command: make prepare-data")
        return
        
    if not summary_data:
        print("Recommendation:")
        print("  -> Run the 3-seed paper-memory suite to perform the core experiments.")
        print("  Command: make paper-memory")
        return
        
    # Analyze summary data
    print("Current Experiment Results Summary:")
    print(f"  Seeds evaluated: {summary_data.get('seeds', 'Unknown')}")
    
    # We will try to read the paper gates report if generated, or evaluate G1-G4
    print("\nNext Steps:")
    print("  -> Review the aggregate summary and run gate evaluation:")
    print("  Command: make paper-memory-evaluate-gates")
    
    # If G3 is not run, run external quality evaluation
    print("  -> Run external evaluation (ARC, HellaSwag, etc.) to pass G3:")
    print("  Command: make paper-memory-external-eval")

def main():
    data_ok = check_datasets()
    summary_data = check_experiment_runs()
    evaluate_and_suggest(data_ok, summary_data)

if __name__ == "__main__":
    main()
