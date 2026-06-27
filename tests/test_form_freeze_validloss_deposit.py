"""Tests for ``scripts/form_freeze_validloss_deposit.py`` — the turnkey that
collapses recipe TASK-0152 Tier-1 step 3 (manual ``best_valid_loss``
transcription) into one reproducible command.

The §4 verdict gate (:mod:`scripts.replay_freeze_validloss_ci`) is fully ready
to judge a real ``proxy_scale: false`` 9B deposit; what is missing is the
*formation* of that deposit from the upstream training artifacts
(``run_metrics.jsonl``). Hand-transcribing ``best_valid_loss`` into a deposit is
a P0 reproducibility hazard (a mistyped float silently corrupts the verdict and
leaves no provenance back to the run). This module reads the artifact directly,
records which ``run_id`` / seed / step each float came from, and emits the exact
schema the judge reads — so the deposit→verdict path carries real numbers
deterministically and auditably.

The suite guards:

* **Extraction** — ``best_valid_loss`` is read from the ``run_footer`` line when
  present (the durable recorded field) and falls back to the min ``loss_valid``
  over ``step`` lines for runs that predate the footer.
* **Schema + provenance** — the formed deposit carries ``candidate_losses`` /
  ``surrogate_losses`` arrays, ``proxy_scale=false`` / ``synthetic=false`` /
  ``negative_control=false`` (so it is NOT a proxy or plumbing recording), and a
  ``source`` string naming every contributing ``run_id``.
* **End-to-end through the judge** — a formed deposit with a clear
  candidate-beats-surrogate separation replays as ``SURPASSES`` and opens the
  ``citable_as_target_scale`` gate (the contract a genuine 9B run earns). This
  pins the whole form→judge chain, not just the formatter.
* **CLI health** — the script launches as ``-m`` (the canary every
  ``scripts.*`` CLI in this repo keeps) and writes a judge-ready file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.form_freeze_validloss_deposit import (
    extract_best_valid_loss,
    form_deposit,
    main,
)
from scripts.replay_freeze_validloss_ci import replay_samples, replay_to_json
from src.tg_lora.freeze_surrogate_gate import SURPASSES


def _write_run_metrics(
    path: Path,
    *,
    run_id: str,
    seed: int,
    best: float,
    step_losses: list[float],
    model: str = "Qwen/Qwen3.5-9B",
    with_footer: bool = True,
) -> Path:
    """Write a minimal ``run_metrics.jsonl`` honoring the real schema fields."""
    lines: list[str] = [
        json.dumps(
            {
                "type": "run_header",
                "run_id": run_id,
                "model_name": model,
                "seed": seed,
                "config_path": "configs/9b_tg_lora.yaml",
            }
        )
    ]
    for i, lv in enumerate(step_losses):
        lines.append(
            json.dumps(
                {
                    "type": "step",
                    "run_id": run_id,
                    "step": (i + 1) * 24,
                    "cycle": i,
                    "loss_valid": lv,
                }
            )
        )
    if with_footer:
        lines.append(
            json.dumps(
                {
                    "type": "run_footer",
                    "run_id": run_id,
                    "best_valid_loss": best,
                    "best_valid_step": 48,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def test_extract_prefers_run_footer_field(tmp_path: Path) -> None:
    p = _write_run_metrics(
        tmp_path / "c0.jsonl",
        run_id="cand_seed42",
        seed=42,
        best=1.03,
        step_losses=[1.5, 1.1, 1.2],
    )
    value, prov = extract_best_valid_loss(p)
    assert value == pytest.approx(1.03)
    assert prov["run_id"] == "cand_seed42"
    assert prov["seed"] == 42
    assert prov["model_name"] == "Qwen/Qwen3.5-9B"
    assert prov["best_valid_step"] == 48
    assert prov["best_valid_loss_source"] == "run_footer"
    assert prov["source"] == str(p)


def test_extract_falls_back_to_min_step_when_no_footer(tmp_path: Path) -> None:
    # min loss_valid across steps is 1.05 (index 1)
    p = _write_run_metrics(
        tmp_path / "nofb.jsonl",
        run_id="b1",
        seed=1,
        best=1.05,  # ignored — no footer written
        step_losses=[1.4, 1.05, 1.3],
        with_footer=False,
    )
    value, prov = extract_best_valid_loss(p)
    assert value == pytest.approx(1.05)
    # fallback records the step the min loss occurred at (index 1 => step 48)
    assert prov["best_valid_step"] == 48
    assert prov["best_valid_loss_source"] == "min_loss_valid_step"


def test_form_deposit_emits_target_scale_schema(tmp_path: Path) -> None:
    cand: list[Path] = []
    base: list[Path] = []
    for seed, best in [(42, 1.03), (43, 1.06), (44, 1.01)]:
        cand.append(
            _write_run_metrics(
                tmp_path / f"c{seed}.jsonl",
                run_id=f"cand_{seed}",
                seed=seed,
                best=best,
                step_losses=[1.5, best, 1.2],
            )
        )
    for seed, best in [(42, 1.12), (43, 1.10), (44, 1.14)]:
        base.append(
            _write_run_metrics(
                tmp_path / f"b{seed}.jsonl",
                run_id=f"base_{seed}",
                seed=seed,
                best=best,
                step_losses=[1.5, best, 1.3],
            )
        )
    deposit = form_deposit(
        cand,
        base,
        model="Qwen/Qwen3.5-9B",
        device="cuda-rtx3060",
        task="generalize",
        architecture="heterogeneous",
        base_seed=0,
        total=120,
    )
    # input order preserved; floats read straight from the artifacts
    assert deposit["candidate_losses"] == [1.03, 1.06, 1.01]
    assert deposit["surrogate_losses"] == [1.12, 1.10, 1.14]
    # genuine target-scale recording — NOT a proxy, synthetic, or negative control
    assert deposit["proxy_scale"] is False
    assert deposit["synthetic"] is False
    assert deposit["negative_control"] is False
    assert deposit["model"] == "Qwen/Qwen3.5-9B"
    assert deposit["device"] == "cuda-rtx3060"
    assert deposit["task"] == "generalize"
    assert deposit["architecture"] == "heterogeneous"
    assert deposit["n_candidate"] == 3
    assert deposit["n_surrogate"] == 3
    assert deposit["base_seed"] == 0
    # symmetric per-arm budget => not a degraded-arm negative control
    assert (
        deposit["candidate_total"]
        == deposit["surrogate_total"]
        == deposit["total"]
        == 120
    )
    # provenance names every contributing run
    assert "cand_42" in deposit["source"]
    assert "base_44" in deposit["source"]


def test_deposit_replays_as_citable_target_scale(tmp_path: Path) -> None:
    cand: list[Path] = []
    base: list[Path] = []
    for seed, best in [(42, 1.00), (43, 0.99), (44, 1.01)]:
        cand.append(
            _write_run_metrics(
                tmp_path / f"c{seed}.jsonl",
                run_id=f"c{seed}",
                seed=seed,
                best=best,
                step_losses=[best],
            )
        )
    for seed, best in [(42, 1.20), (43, 1.21), (44, 1.19)]:
        base.append(
            _write_run_metrics(
                tmp_path / f"b{seed}.jsonl",
                run_id=f"b{seed}",
                seed=seed,
                best=best,
                step_losses=[best],
            )
        )
    deposit = form_deposit(
        cand,
        base,
        model="Qwen/Qwen3.5-9B",
        device="cuda",
        task="generalize",
        architecture="heterogeneous",
        base_seed=0,
        total=60,
    )
    ci = replay_samples(deposit)
    out = replay_to_json("<formed>", deposit, ci)
    assert out["proxy_scale"] is False
    assert out["synthetic"] is False
    # a genuine target-scale recording opens the citation gate (the §4 contract)
    assert out["citable_as_target_scale"] is True
    assert out["replayed_verdict"] == SURPASSES


def test_cli_help_canary_and_file_output(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.form_freeze_validloss_deposit", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "candidate" in proc.stdout.lower()

    c = _write_run_metrics(
        tmp_path / "c.jsonl", run_id="c", seed=42, best=1.0, step_losses=[1.0]
    )
    b = _write_run_metrics(
        tmp_path / "b.jsonl", run_id="b", seed=42, best=1.3, step_losses=[1.3]
    )
    out = tmp_path / "deposit.json"
    rc = main(
        [
            "--candidate",
            str(c),
            "--surrogate",
            str(b),
            "--model",
            "M",
            "--device",
            "cuda",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["candidate_losses"] == [1.0]
    assert d["surrogate_losses"] == [1.3]
    assert d["proxy_scale"] is False
    assert d["synthetic"] is False


def test_form_deposit_rejects_empty_arm(tmp_path: Path) -> None:
    cand = [
        _write_run_metrics(
            tmp_path / "c.jsonl", run_id="c", seed=42, best=1.0, step_losses=[1.0]
        )
    ]
    # no surrogate runs => cannot form a comparison deposit
    with pytest.raises((ValueError, SystemExit)):
        form_deposit(cand, [], model="M", device="cuda", base_seed=0)
