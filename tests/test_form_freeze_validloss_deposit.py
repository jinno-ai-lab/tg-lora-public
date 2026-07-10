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
    progressive_freeze: dict | None = None,
) -> Path:
    """Write a minimal ``run_metrics.jsonl`` honoring the real schema fields.

    ``progressive_freeze`` (the block ``progressive_freeze_run_summary`` emits —
    ``policy`` / ``resolved_policy`` / ``surrogate_seed`` / ``mode``) is embedded
    under ``tg_lora_summary`` exactly as ``train_tg_lora``'s footer writes it,
    so the Tier-2 arm-identity extraction is exercised against the real shape.
    """
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
        footer: dict = {
            "type": "run_footer",
            "run_id": run_id,
            "best_valid_loss": best,
            "best_valid_step": 48,
        }
        if progressive_freeze is not None:
            footer["tg_lora_summary"] = {"progressive_freeze": progressive_freeze}
        lines.append(json.dumps(footer))
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


def test_extract_falls_back_when_footer_best_is_null(tmp_path: Path) -> None:
    # An interrupted run writes a footer with best_valid_loss: null (best_loss
    # never updated) but still carries per-step loss_valid lines. Extract must
    # fall back to the step minimum rather than crash on float(None).
    p = tmp_path / "killed.jsonl"
    lines = [
        json.dumps(
            {
                "type": "run_header",
                "run_id": "killed_base",
                "model_name": "Qwen/Qwen3.5-9B",
                "seed": 42,
            }
        ),
        # per-step loss_valid present (min = 1.06 at index 1)
        json.dumps({"type": "step", "step": 8, "loss_valid": 1.30}),
        json.dumps({"type": "step", "step": 16, "loss_valid": 1.06}),
        json.dumps({"type": "step", "step": 24, "loss_valid": 1.18}),
        # footer exists but best_valid_loss is null (never updated before kill)
        json.dumps(
            {"type": "run_footer", "best_valid_loss": None, "best_valid_step": 0}
        ),
    ]
    p.write_text("\n".join(lines) + "\n")
    value, prov = extract_best_valid_loss(p)
    assert value == pytest.approx(1.06)
    assert prov["best_valid_loss_source"] == "min_loss_valid_step"
    assert prov["best_valid_step"] == 16
    assert prov["run_id"] == "killed_base"


def _real_producer_metrics(tmp_path: Path, run_id: str, *, with_footer: bool = False):
    """A REAL ``RunMetrics`` producer with a minimal but real ``run_header``.

    Real interrupted runs always write their header early (``write_header`` runs
    before any step), so a faithful reproduction emits one rather than relying
    on the ``path.stem`` fallback for ``run_id`` provenance.
    """
    from types import SimpleNamespace

    from src.utils.run_metrics import RunMetrics

    cfg = SimpleNamespace(
        model=SimpleNamespace(name_or_path="Qwen/Qwen3.5-9B"),
        lora=SimpleNamespace(r=8, alpha=16),
        training=SimpleNamespace(
            batch_size=1,
            grad_accumulation=4,
            learning_rate=1e-4,
            optimizer_lifecycle="recreate_per_cycle",
        ),
        experiment=SimpleNamespace(seed=42),
        tg_lora=None,
        alpha_line=None,
        _path="configs/9b_tg_lora.yaml",
    )
    m = RunMetrics(tmp_path, mode="tg_lora", run_id=run_id)
    m.write_header(cfg, budget_type="backward_passes", budget_value=10)
    return m, cfg


def test_extract_fallback_uses_honest_full_eval_when_lower(tmp_path: Path) -> None:
    # The producer's ``cycle_state.best_loss`` (→ footer ``best_valid_loss``) is
    # updated by BOTH the pilot proxy (``record_cycle`` → ``loss_valid``) AND
    # the honest full-eval (``record_full_eval`` → ``loss_valid_full``). An
    # interrupted run (no footer) that emitted an honest ``loss_valid_full``
    # record BELOW the proxy minimum must surface that honest loss — otherwise
    # the deposit misrepresents what the footer would have recorded. Uses the
    # REAL ``RunMetrics`` producer (real header + step + full-eval records) so
    # the defect is reproduced from the actual emission path.
    m, _ = _real_producer_metrics(tmp_path, "killed_with_fulleval")
    try:
        # pilot-proxy losses per cycle (proxy min = 1.4 at step 48)
        m.record_step(
            step=24, cycle=0, loss_train=2.0, loss_valid=1.5, total_backward_passes=24
        )
        m.record_step(
            step=48, cycle=1, loss_train=1.9, loss_valid=1.4, total_backward_passes=48
        )
        # honest full-eval (1.2 < proxy min 1.4): loss_valid=None, loss_valid_full=1.2
        m.record_full_eval_loss(
            cycle=1, full_loss=1.2, total_backward_passes=48, step=48
        )
        # NO write_footer — the run was interrupted/killed before completion.
    finally:
        m.close()

    value, prov = extract_best_valid_loss(tmp_path / "run_metrics.jsonl")
    assert value == pytest.approx(1.2)  # honest full-eval, NOT proxy min 1.4
    assert prov["best_valid_loss_source"] == "min_loss_valid_step"
    assert prov["best_valid_step"] == 48
    assert prov["run_id"] == "killed_with_fulleval"


def test_extract_fallback_takes_true_min_across_proxy_and_full_eval(
    tmp_path: Path,
) -> None:
    # Guard against an over-correction that ALWAYS prefers ``loss_valid_full``:
    # the fallback must take the true min across both fields (matching the
    # producer's ``best_loss = min(proxy, full_eval)``). When the pilot proxy is
    # the lower value, it wins. Real producer output.
    m, _ = _real_producer_metrics(tmp_path, "proxy_lower")
    try:
        m.record_step(
            step=24, cycle=0, loss_train=2.0, loss_valid=1.1, total_backward_passes=24
        )
        m.record_step(
            step=48, cycle=1, loss_train=1.9, loss_valid=1.3, total_backward_passes=48
        )
        # honest full-eval (1.2) is HIGHER than the proxy min (1.1)
        m.record_full_eval_loss(
            cycle=1, full_loss=1.2, total_backward_passes=48, step=48
        )
    finally:
        m.close()

    value, prov = extract_best_valid_loss(tmp_path / "run_metrics.jsonl")
    assert value == pytest.approx(1.1)  # proxy is the true min (1.1 < honest 1.2)
    assert prov["best_valid_step"] == 24


def test_extract_surfaces_progressive_freeze_arm_provenance(tmp_path: Path) -> None:
    # A Tier-2 surrogate (random_order) run: the footer's progressive_freeze
    # block carries the requested policy + surrogate seed, the only
    # machine-readable arm distinguisher at full freeze depth.
    p = _write_run_metrics(
        tmp_path / "surr_seed42.jsonl",
        run_id="surr_seed42",
        seed=42,
        best=1.04,
        step_losses=[1.2, 1.04],
        progressive_freeze={
            "enabled": True,
            "frozen_layers": [0, 1, 2],
            "start_cycle": 3,
            "mode": "progressive",
            "policy": "random_order",
            "resolved_policy": "convergence_order",
            "surrogate_seed": 42,
            "realized_depth": 3,
        },
    )
    _, prov = extract_best_valid_loss(p)
    assert prov["arm_policy"] == "random_order"
    assert prov["arm_resolved_policy"] == "convergence_order"
    assert prov["surrogate_seed"] == 42
    assert prov["freeze_mode"] == "progressive"


def test_extract_arm_provenance_none_without_progressive_freeze(tmp_path: Path) -> None:
    # A Tier-1 candidate (plain TG-LoRA, no progressive-freeze footer block)
    # reports no arm identity — the extraction stays graceful so the Tier-1
    # deposit path is unaffected.
    p = _write_run_metrics(
        tmp_path / "cand.jsonl",
        run_id="cand",
        seed=42,
        best=1.03,
        step_losses=[1.2, 1.03],
    )
    _, prov = extract_best_valid_loss(p)
    assert prov["arm_policy"] is None
    assert prov["surrogate_seed"] is None
    assert prov["freeze_mode"] is None


def test_arm_label_includes_policy_for_progressive_arm(tmp_path: Path) -> None:
    p = _write_run_metrics(
        tmp_path / "c.jsonl",
        run_id="cand_seed42",
        seed=42,
        best=1.03,
        step_losses=[1.03],
        progressive_freeze={
            "mode": "progressive",
            "policy": "output_first",
            "resolved_policy": "output_first",
            "surrogate_seed": None,
            "realized_depth": 2,
        },
    )
    from scripts.form_freeze_validloss_deposit import _arm_label

    _, prov = extract_best_valid_loss(p)
    label = _arm_label(1.03, prov)
    assert "arm=output_first" in label
    assert "cand_seed42" in label


def test_form_deposit_accepts_aligned_tier2_arms(tmp_path: Path) -> None:
    # candidate = output_first (real arm, surrogate_seed None);
    # surrogate = random_order (surrogate_seed set). Both multi-layer
    # progressive => guard verifies and accepts the aligned assignment.
    cand = _write_run_metrics(
        tmp_path / "c.jsonl",
        run_id="c",
        seed=42,
        best=1.03,
        step_losses=[1.03],
        progressive_freeze={
            "mode": "progressive",
            "policy": "output_first",
            "resolved_policy": "output_first",
            "surrogate_seed": None,
            "realized_depth": 2,
        },
    )
    surr = _write_run_metrics(
        tmp_path / "s.jsonl",
        run_id="s",
        seed=42,
        best=1.10,
        step_losses=[1.10],
        progressive_freeze={
            "mode": "progressive",
            "policy": "random_order",
            "resolved_policy": "convergence_order",
            "surrogate_seed": 7,
            "realized_depth": 2,
        },
    )
    deposit = form_deposit(
        [cand],
        [surr],
        model="Qwen/Qwen3.5-9B",
        device="cuda-rtx3060",
        total=15,
    )
    assert deposit["candidate_losses"] == [1.03]
    assert deposit["surrogate_losses"] == [1.10]
    # machine-readable arm identity flows footer -> deposit
    assert deposit["candidate_arm_policies"] == ["output_first"]
    assert deposit["surrogate_arm_policies"] == ["random_order"]


def test_form_deposit_rejects_surrogate_passed_as_candidate(tmp_path: Path) -> None:
    # A random-order surrogate (seed set) mistakenly passed under --candidate.
    # Without the guard this silently deposits a surrogate float in the
    # candidate slot and inverts the verdict sign.
    swapped = _write_run_metrics(
        tmp_path / "s.jsonl",
        run_id="s",
        seed=42,
        best=1.10,
        step_losses=[1.10],
        progressive_freeze={
            "mode": "progressive",
            "policy": "random_order",
            "resolved_policy": "convergence_order",
            "surrogate_seed": 7,
            "realized_depth": 2,
        },
    )
    real = _write_run_metrics(
        tmp_path / "c.jsonl",
        run_id="c",
        seed=42,
        best=1.03,
        step_losses=[1.03],
        progressive_freeze={
            "mode": "progressive",
            "policy": "output_first",
            "resolved_policy": "output_first",
            "surrogate_seed": None,
            "realized_depth": 2,
        },
    )
    with pytest.raises(ValueError, match="random-order surrogate"):
        form_deposit(
            [swapped],
            [real],
            model="Qwen/Qwen3.5-9B",
            device="cuda-rtx3060",
            total=15,
        )


def test_form_deposit_rejects_real_arm_passed_as_surrogate(tmp_path: Path) -> None:
    # Symmetric: a real arm (output_first, seed None) passed under --surrogate
    # where a random-order surrogate belongs. The candidate slot holds a proper
    # real arm; the surrogate slot wrongly holds another real arm.
    cand = _write_run_metrics(
        tmp_path / "c.jsonl",
        run_id="c",
        seed=42,
        best=1.03,
        step_losses=[1.03],
        progressive_freeze={
            "mode": "progressive",
            "policy": "output_first",
            "resolved_policy": "output_first",
            "surrogate_seed": None,
            "realized_depth": 2,
        },
    )
    real_as_surrogate = _write_run_metrics(
        tmp_path / "s.jsonl",
        run_id="s",
        seed=43,
        best=1.10,
        step_losses=[1.10],
        progressive_freeze={
            "mode": "progressive",
            "policy": "output_first",
            "resolved_policy": "output_first",
            "surrogate_seed": None,
            "realized_depth": 2,
        },
    )
    with pytest.raises(ValueError, match="was passed as --surrogate"):
        form_deposit(
            [cand],
            [real_as_surrogate],
            model="Qwen/Qwen3.5-9B",
            device="cuda-rtx3060",
            total=15,
        )


def test_form_deposit_skips_arm_guard_for_tier1_no_progressive(tmp_path: Path) -> None:
    # Tier-1: candidate = plain TG-LoRA, surrogate = full-backprop baseline.
    # Neither carries a progressive_freeze footer block, so the guard has
    # nothing to verify and must stay silent (regression guard for the Tier-1
    # deposit recipe in TASK-0152).
    cand = _write_run_metrics(
        tmp_path / "c.jsonl", run_id="c", seed=42, best=1.03, step_losses=[1.03]
    )
    base = _write_run_metrics(
        tmp_path / "b.jsonl", run_id="b", seed=42, best=1.10, step_losses=[1.10]
    )
    deposit = form_deposit(
        [cand],
        [base],
        model="Qwen/Qwen3.5-9B",
        device="cuda-rtx3060",
        total=15,
    )
    assert deposit["candidate_arm_policies"] == [None]
    assert deposit["surrogate_arm_policies"] == [None]


def test_form_deposit_skips_arm_guard_for_single_shot_mode(tmp_path: Path) -> None:
    # A single-shot progressive-freeze footer (mode=single_shot, Phase 1 gate)
    # carries a progressive_freeze block but no verifiable multi-layer arm
    # identity — the guard must skip it, not reject it.
    cand = _write_run_metrics(
        tmp_path / "c.jsonl",
        run_id="c",
        seed=42,
        best=1.03,
        step_losses=[1.03],
        progressive_freeze={
            "enabled": True,
            "frozen_layers": [2],
            "start_cycle": 3,
            "mode": "single_shot",
        },
    )
    base = _write_run_metrics(
        tmp_path / "b.jsonl", run_id="b", seed=42, best=1.10, step_losses=[1.10]
    )
    deposit = form_deposit(
        [cand],
        [base],
        model="Qwen/Qwen3.5-9B",
        device="cuda-rtx3060",
        total=15,
    )
    # single_shot arm carries no requested policy => None, not rejected
    assert deposit["candidate_arm_policies"] == [None]


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


# ---------------------------------------------------------------------------
# Reduced-context provenance tag (the seq_len=256 probe honesty axis)
# ---------------------------------------------------------------------------


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    c = _write_run_metrics(tmp_path / "c.jsonl", run_id="c", seed=42, best=1.0, step_losses=[1.0])
    b = _write_run_metrics(tmp_path / "b.jsonl", run_id="b", seed=42, best=1.3, step_losses=[1.3])
    return c, b


@pytest.mark.parametrize(
    "seq_len, full_context, expected",
    [
        (1024, False, True),   # seq_len>=1024 marks the full §4 verdict
        (256, False, False),   # seq_len=256 stays reduced-context
        (None, True, True),    # explicit --full-context override
    ],
)
def test_form_deposit_context_tag(
    tmp_path: Path, seq_len: int | None, full_context: bool, expected: bool
) -> None:
    # The form CLI tags the deposit's context: only seq_len>=1024 or an explicit
    # --full-context marks it the full §4 verdict — the over-cite a 12GB seq256
    # probe could otherwise invite is impossible by construction (TASK-0152 86-97).
    c, b = _pair(tmp_path)
    deposit = form_deposit(
        [c], [b], model="M", device="cuda", seq_len=seq_len, full_context=full_context
    )
    assert deposit["full_context"] is expected
    assert deposit["seq_len"] == seq_len


def test_default_deposit_is_reduced_context(tmp_path: Path) -> None:
    # No flags => the honest 12GB-GPU assumption: reduced-context. A default of
    # True here is exactly the over-cite the guard exists to prevent.
    c, b = _pair(tmp_path)
    deposit = form_deposit([c], [b], model="M", device="cuda", base_seed=0)
    assert deposit["full_context"] is False
    assert deposit["seq_len"] is None


def test_reduced_deposit_replays_target_scale_but_not_full_verdict(tmp_path: Path) -> None:
    # End-to-end honesty contract: a default (reduced-context) deposit opens
    # citable_as_target_scale (real 9B run) but NOT citable_as_full_section4_verdict.
    c, b = _pair(tmp_path)
    deposit = form_deposit([c], [b], model="M", device="cuda")
    out = replay_to_json("<formed>", deposit, replay_samples(deposit))
    assert out["citable_as_target_scale"] is True
    assert out["citable_as_full_section4_verdict"] is False


def test_cli_seq_len_and_full_context_flags(tmp_path: Path) -> None:
    # --seq-len 1024 marks the deposit full; the default (no flag) stays reduced.
    c, b = _pair(tmp_path)
    full = tmp_path / "full.json"
    assert main(["--candidate", str(c), "--surrogate", str(b), "--model", "M",
                 "--device", "cuda", "--seq-len", "1024", "--output", str(full)]) == 0
    assert json.loads(full.read_text())["full_context"] is True
    reduced = tmp_path / "reduced.json"
    assert main(["--candidate", str(c), "--surrogate", str(b), "--model", "M",
                 "--device", "cuda", "--output", str(reduced)]) == 0
    assert json.loads(reduced.read_text())["full_context"] is False
