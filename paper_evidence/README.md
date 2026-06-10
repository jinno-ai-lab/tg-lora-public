# Paper Evidence Directory

This directory stores the official empirical evidence referenced in the TG-LoRA manuscript. To keep the repository size manageable, all large checkpoints and binary weights (`*.safetensors`, `*.pt`, `*.bin`, `*.gguf`) are excluded, preserving only the lightweight configuration files, metrics logs, reports, and execution statistics.

---

## 1. Directory Structure

- `paper_memory_one_shot_suite/`: Contains configurations, training metrics logs (`run_metrics.jsonl`), and step summaries for the 3-seed One-Shot cache comparisons (Seeds 42, 43, 44).
- `frontier_sweep/`: Contains memory frontier sweep logs and exit codes for sequence lengths 1536, 2048, and 3072 under Seed 42.

Each folder contains a `version_metadata.json` capturing the exact git commit SHA1 and the presence of any uncommitted local differences when the metrics were produced.

---

## 2. Guidelines for Updating Evidence

When updating the evidence (e.g. after the 3-seed downstream evaluations finish in the `runs/` folder), use `rsync` to ingest the new files while strictly filtering out heavy binaries:

```bash
# Ingest One-Shot Suite runs (excluding model checkpoints/weights)
rsync -av \
  --exclude="*.safetensors" \
  --exclude="*.bin" \
  --exclude="*.pt" \
  --exclude="*.pth" \
  --exclude="*.gguf" \
  --exclude="checkpoint-*/" \
  runs/paper_memory_one_shot_suite_20260531_192119/ \
  paper_evidence/paper_memory_one_shot_suite/

# Ingest Frontier Sweep runs
rsync -av \
  --exclude="*.safetensors" \
  --exclude="*.bin" \
  --exclude="*.pt" \
  --exclude="*.pth" \
  --exclude="*.gguf" \
  --exclude="checkpoint-*/" \
  runs/frontier_sweep/ \
  paper_evidence/frontier_sweep/
```

After updating, verify the total size of the `paper_evidence/` folder remains minimal (typically < 300MB):
```bash
du -sh paper_evidence/
find paper_evidence/ -name "*.pt" -o -name "*.safetensors"
```
