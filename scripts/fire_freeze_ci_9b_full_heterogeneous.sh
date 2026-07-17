#!/usr/bin/env bash
# Robust launcher for the FULL-BUDGET × HETEROGENEOUS 9B §4 verdict — the ONE
# remaining open research leg (homogeneous full LANDED→TIES ``4b88ca8``;
# heterogeneous REDUCED-budget LANDED→SURPASSES ``db542fe``/``d00a362``; whether
# the heterogeneous SURPASSES survives the full 1500-step budget on an asymmetric
# per-layer-rank adapter is unmeasured).
#
# This is the committed, version-controlled analogue of the ad-hoc
# ``/home/jinno/tg-lora-public-full-run/fire.sh`` that robustly fired the
# HOMOGENEOUS full run. Everything resolves OUTSIDE any AI-Hub worktree, so
# recycling the worktree mid-run cannot break the per-arm
# ``-m scripts.run_freeze_validloss_ci_9b`` import (modules resolved via PYTHONPATH
# + stable CWD) nor lose the deposit (``--ledger``/``--output``/``--config`` are
# absolute). The bg launcher polls a held GPU (defers exit 75 until ~10 GiB free =
# after any concurrent run), banks each completed arm in ``--ledger``, and is
# bounded (``--max-attempts`` / ``--deadline-seconds``) so it cannot spin forever.
#
# Arm shape mirrors the Makefile ``FREEZE_9B_FULL_HETEROGENEOUS_FLAGS``: candidate
# (output-first suffix {29,30,31}) + surrogate (random order) + input-side
# control (input-first contiguous {24,25,26}) = the direction-isolation A/B,
# bumped 96 → 1500 steps; ``--total-steps 1500`` reaches the config max_steps so
# ``reduced_budget=False``, and ``--train-examples 600`` keeps a 1500-step run at
# ~2.5 epochs = generalization regime (the 4th honesty axis a naive 1500/48 run
# would violate by memorizing). Distinct deposit + ledger so the heterogeneous
# full verdict never clobbers the homogeneous full deposit/ledger.
#
# Prep (once, after a code change): re-sync the stable repo's code from the
# worktree so the harness has current heterogeneous support:
#   rsync -a --delete scripts/ src/ configs/ \
#     /home/jinno/tg-lora-public-full-run/repo/{scripts,src,configs}/  # (expand)
#
# Fire detached:
#   nohup bash scripts/fire_freeze_ci_9b_full_heterogeneous.sh \
#     > /home/jinno/tg-lora-public-full-run/full_heterogeneous_bg.log 2>&1 &
#
# Harvest (next session, when the ledger has 9 lines + the deposit is written).
# This verdict is the 2nd citable full-budget result, so it earns the SAME
# independent-reproducibility provenance the homogeneous full got at ``7489023``:
# a committed LEDGER WITNESS. The worker stamps ``run_log_*`` but NEVER
# ``ledger_witness_*`` — those are HARVEST fields stamped post-hoc into the
# deposit JSON. Forgetting the witness would land a verdict that is citable but
# NOT independently reproducible from committed bytes (the exact gap ``7489023``
# closed); ``TestCommittedLedgerWitnessHeterogeneousFull`` auto-verifies the
# harvest below, so an incomplete harvest fails the test suite rather than
# passing silently. Full harvest (run from the repo root):
#   # 1. committed deposit + ledger witness:
#   cp /home/jinno/tg-lora-public-full-run/freeze_validloss_ci_9b_full_heterogeneous.json \
#     tests/fixtures/
#   cp /home/jinno/tg-lora-public-full-run/runs/freeze_validloss_ci_9b_full_heterogeneous_ledger.jsonl \
#     tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous_ledger.jsonl
#   # 2. stamp the HARVEST witness fields into the deposit JSON (relative path +
#   #    canonical SHA-256 over the parsed JSONL, same canonicalization as
#   #    TestCommittedLedgerWitness._ledger_witness_sha256):
#   python -c "
#   import json, hashlib
#   p='tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous.json'
#   L='tests/fixtures/freeze_validloss_ci_9b_full_heterogeneous_ledger.jsonl'
#   d=json.load(open(p))
#   parsed=[json.loads(l) for l in open(L).read().splitlines() if l.strip()]
#   d['ledger_witness_path']=L
#   d['ledger_witness_sha256']=hashlib.sha256(
#       json.dumps(parsed,sort_keys=True,separators=(',',':')).encode()).hexdigest()
#   json.dump(d,open(p,'w'),indent=2,sort_keys=False); print(d['ledger_witness_sha256'])
#   "
#   # 3. pin the two frozen literals the test suite forces you to set on a real
#   #    deposit: TestCommittedLedgerWitnessHeterogeneousFull._FROZEN_WITNESS_HASH
#   #    and the _EXPECTED entry in TestDepositEvidenceHash (both currently skip).
# Do NOT re-fire the homogeneous leg — it is harvested (``4b88ca8``).
set -euo pipefail

# Machine-specific paths — overridable via env for portability / re-targeting.
STABLE="${TG_LORA_FULL_RUN_DIR:-/home/jinno/tg-lora-public-full-run}"
PY="${PYTHON_VENV:-/home/jinno/tg-lora/.venv/bin/python}"   # torch + peft + bnb + datasets + omegaconf
LAUNCH_FLAGS="${LAUNCH_FLAGS:-}"                              # e.g. "--max-attempts 600 --tempfail-sleep 120"

cd "$STABLE/repo"                                            # stable CWD; scripts/ + src/ importable
export PYTHONPATH="$STABLE/repo:${PYTHONPATH:-}"
export PYTHON_VENV="$PY"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # 12 GiB-card headroom

exec "$PY" -m scripts.launch_freeze_ci_9b_full $LAUNCH_FLAGS -- \
    --architecture heterogeneous \
    --seq-len 1024 --total-steps 1500 --warmup-steps 150 --depth 3 --spacing 450 \
    --n-candidate 3 --n-surrogate 3 --n-control 3 \
    --train-examples 600 --valid-examples 64 --max-dataset-rows 2000 \
    --config "$STABLE/repo/configs/9b_baseline_suffix_only_last25.yaml" \
    --ledger "$STABLE/runs/freeze_validloss_ci_9b_full_heterogeneous_ledger.jsonl" \
    --json --output "$STABLE/freeze_validloss_ci_9b_full_heterogeneous.json"
