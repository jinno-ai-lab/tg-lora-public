#!/usr/bin/env python
import os
import shutil
import sys
from pathlib import Path
import yaml

# Add scripts directory to path to allow imports
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

from git_utils import save_git_metadata  # noqa: E402

# Define allowed extensions for evidence
ALLOWED_EXTENSIONS = {'.json', '.jsonl', '.yaml', '.yml', '.md', '.txt', '.png'}
ALLOWED_FILENAMES = {'exit_code'}

def is_evidence_file(file_path: Path) -> bool:
    """Check if the file is an allowed evidence file type."""
    if file_path.suffix.lower() in ALLOWED_EXTENSIONS:
        return True
    if file_path.name in ALLOWED_FILENAMES:
        return True
    return False

def load_config(config_path: Path):
    """Load configuration yaml safely."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load config at {config_path}: {e}")
        return None

def copy_evidence(src_dir: Path, dest_dir: Path):
    """Recursively copy only allowed evidence files from src_dir to dest_dir."""
    for root, dirs, files in os.walk(src_dir):
        # Exclude checkpoint and temporary directories from scanning
        if 'checkpoint-' in Path(root).name:
            continue
            
        rel_path = Path(root).relative_to(src_dir)
        target_root = dest_dir / rel_path
        
        for file in files:
            src_file = Path(root) / file
            if is_evidence_file(src_file):
                # Ensure target directory exists
                target_root.mkdir(parents=True, exist_ok=True)
                dest_file = target_root / file
                
                # Perform copy (and overwrite if changed)
                shutil.copy2(src_file, dest_file)

def main():
    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs"
    evidence_dir = repo_root / "paper_evidence"
    
    if not runs_dir.exists():
        print("No runs/ directory found. Nothing to ingest.")
        return

    evidence_dir.mkdir(parents=True, exist_ok=True)
    ingested_count = 0
    
    print("Scanning runs/ for paper-flagged experiments...")
    
    # Scan all directories inside runs/
    for item in sorted(runs_dir.iterdir()):
        if not item.is_dir():
            continue
            
        # We look for config.yaml at the root of the run folder, or inside sub-configs
        config_candidates = [
            item / "config.yaml",
            item / "seed_42/configs/tg_cache.yaml",
            item / "seed_42/configs/baseline.yaml",
            item / "slen_1536/seed_42/tg_lora/config.yaml"
        ]
        
        # Also recursively find any config.yaml in the top 3 levels to be robust
        found_config = None
        for cand in config_candidates:
            if cand.exists():
                found_config = cand
                break
                
        if not found_config:
            # Fallback recursive search for config.yaml up to depth 3
            for p in item.glob("**/config.yaml"):
                found_config = p
                break
                
        if not found_config:
            continue
            
        config = load_config(found_config)
        if not config or 'experiment' not in config:
            continue
            
        exp_meta = config['experiment']
        is_paper = exp_meta.get('paper_experiment', False)
        exp_id = exp_meta.get('paper_experiment_id', None)
        
        if is_paper and exp_id:
            dest_folder = evidence_dir / exp_id
            print(f"\n[FOUND] Paper experiment run: '{item.name}' -> ID: '{exp_id}'")
            print(f"        Ingesting evidence files into: {dest_folder.relative_to(repo_root)}")
            
            # Copy matching files
            copy_evidence(item, dest_folder)
            
            # Write Git version metadata to the ingested folder
            save_git_metadata(dest_folder)
            ingested_count += 1
            
    print(f"\nIngestion finished. Successfully processed {ingested_count} paper-flagged runs.")

if __name__ == "__main__":
    main()
