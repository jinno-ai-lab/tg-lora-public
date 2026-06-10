import subprocess
import json
from pathlib import Path

def get_git_metadata():
    """Retrieve current git commit SHA1 and check for any local differences."""
    metadata = {
        "git_commit_sha1": "unknown",
        "has_local_changes": False,
        "local_diff": ""
    }
    
    # Path exclusions to prevent massive diff files during evidence copying
    path_exclusions = [
        "--",
        ".",
        ":!paper_evidence/*",
        ":!runs/*"
    ]
    
    try:
        # Get SHA1
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        metadata["git_commit_sha1"] = sha
        
        # Check local changes
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"] + path_exclusions, 
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        
        if status_out:
            metadata["has_local_changes"] = True
            # Get git diff details with path exclusions
            diff_out = subprocess.check_output(
                ["git", "diff"] + path_exclusions, 
                stderr=subprocess.DEVNULL
            ).decode("utf-8")
            
            # Truncate details if they are too large
            MAX_DIFF_CHAR = 50000  # ~50 KB safety limit
            if len(diff_out) > MAX_DIFF_CHAR:
                metadata["local_diff"] = diff_out[:MAX_DIFF_CHAR] + "\n\n... [DIFF TRUNCATED TO 50KB TO PREVENT FILE BLOAT] ...\n"
            else:
                metadata["local_diff"] = diff_out
            
    except Exception as e:
        # Git command failed (e.g. not a git repo or git not installed)
        metadata["error"] = str(e)
        
    return metadata


def save_git_metadata(output_dir):
    """Save the retrieved git metadata to the specified output directory as version_metadata.json."""
    output_path = Path(output_dir) / "version_metadata.json"
    metadata = get_git_metadata()
    
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Git version metadata saved to {output_path}")
    return metadata
