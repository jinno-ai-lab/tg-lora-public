"""Form a GOAL §4 verdict-gate deposit from real upstream ``run_metrics.jsonl``.

Recipe TASK-0152 Tier-1 leaves deposit formation as a manual step: read each
run's ``best_valid_loss`` and type it into a JSON. That transcription is a P0
hazard — a mistyped float silently corrupts the bootstrap verdict and leaves no
trail back to the run that produced it. This script reads the artifact directly
and emits the exact schema :mod:`scripts.replay_freeze_validloss_ci` judges, so
the deposit→verdict path carries real numbers deterministically and auditably.

Usage::

    # candidate = output_first progressive-freeze runs (multi-seed)
    # surrogate = full-backprop baseline runs (multi-seed)
    python -m scripts.form_freeze_validloss_deposit \
        --candidate runs/cand_seed42/run_metrics.jsonl \
                    runs/cand_seed43/run_metrics.jsonl \
                    runs/cand_seed44/run_metrics.jsonl \
        --surrogate runs/base_seed42/run_metrics.jsonl \
                    runs/base_seed43/run_metrics.jsonl \
                    runs/base_seed44/run_metrics.jsonl \
        --model "Qwen/Qwen3.5-9B" --device "cuda-rtx3060" \
        --task generalize --architecture heterogeneous --total 120 \
        --output tests/fixtures/freeze_validloss_9b_target.json

    # then judge it, GPU-free:
    python -m scripts.replay_freeze_validloss_ci tests/fixtures/freeze_validloss_9b_target.json --json

``best_valid_loss`` is read from the durable ``run_footer`` line when present and
falls back to the minimum ``loss_valid`` across ``step`` lines for runs that
predate the footer. The formed deposit carries ``proxy_scale=false`` /
``synthetic=false`` / ``negative_control=false`` (a genuine target-scale
recording, not proxy plumbing) plus a ``source`` string naming every
contributing ``run_id`` / seed — so each deposited float traces back to its
artifact and the §4 citation gate opens only for a real run.

For the Tier-2 §4 order verdict (candidate ``output_first`` vs surrogate
``random_order``, both multi-layer progressive-freeze runs) the footer's
``progressive_freeze`` block is also surfaced: a swapped ``--candidate`` /
``--surrogate`` — which flips the bootstrap sign and turns SURPASSES into
UNDERSHOOTS — fails LOUD at form time rather than corrupting the verdict. The
guard only fires for arms that carry progressive-freeze provenance, so the
Tier-1 candidate (plain TG-LoRA) vs Tier-1 baseline path is unaffected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


def _finite_loss_or_none(value: Any) -> float | None:
    """Coerce a loss field to float, returning ``None`` when non-finite.

    ``float()`` accepts ``"nan"`` / ``"inf"`` / ``"-inf"`` silently, and a
    diverged QLoRA run writes exactly those into ``loss_valid`` /
    ``best_valid_loss`` (gradient explosion → a NaN/inf eval loss). A
    non-finite float deposited into ``candidate_losses`` /
    ``surrogate_losses`` poisons the bootstrap verdict downstream
    (``numpy.mean([1.5, nan]) == nan`` → a NaN CI → a corrupt-but-green §4
    verdict, GOAL §7) — the same silent-corruption class as a non-Dolly
    schema reaching :func:`scripts.run_freeze_validloss_ci_9b._load_dolly_records`
    (``7c4aebf``). Routing a non-finite value through the same ``None``
    "no signal" path as a missing field means a run with an occasional
    glitched eval step still yields its honest finite best, while a
    fully-diverged run (every step AND the footer non-finite) hits the loud
    ``ValueError`` in :func:`extract_best_valid_loss` instead of silently
    depositing NaN.
    """
    f = float(value)
    return f if math.isfinite(f) else None


def extract_best_valid_loss(path: str | Path) -> tuple[float, dict[str, Any]]:
    """Read the best valid loss + provenance from one ``run_metrics.jsonl``.

    Prefers the durable ``run_footer.best_valid_loss`` field; falls back to the
    minimum across ``step`` lines' ``loss_valid`` (pilot proxy) AND
    ``loss_valid_full`` (honest full-eval) for runs that predate the footer, or
    whose footer carries ``best_valid_loss: null`` (an interrupted run that
    never updated ``best_loss``). The producer's ``cycle_state.best_loss``
    tracker — which the footer's ``best_valid_loss`` mirrors — is updated by
    BOTH the pilot proxy (``record_cycle``) and the honest full-eval
    (``record_full_eval``), so the fallback takes the min across both fields to
    stay faithful to what the footer would have recorded (otherwise an
    interrupted run with honest full-eval records would silently surface the
    proxy best). Returns ``(value, provenance)`` where provenance names the run
    so a deposited float always traces back to its source artifact.
    """
    path = Path(path)
    run_id = path.stem
    seed: Any = None
    model_name: Any = None
    footer_value: float | None = None
    footer_step: Any = None
    min_step_loss: float | None = None
    min_step: Any = None
    # Tier-2 §4 order-verdict arm identity from the footer's progressive-freeze
    # block (c28e522): the requested ``policy`` (output_first / random_order) and
    # ``surrogate_seed`` are the only machine-readable distinguisher between the
    # candidate and surrogate arms once both freeze the same layers at full
    # depth. Surfaced here so :func:`form_deposit` can reject a swapped
    # --candidate/--surrogate before it silently inverts the verdict sign.
    arm_policy: Any = None
    arm_resolved_policy: Any = None
    surrogate_seed: Any = None
    freeze_mode: Any = None
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtype = record.get("type")
            if rtype == "run_header":
                run_id = record.get("run_id", run_id)
                seed = record.get("seed", seed)
                model_name = record.get("model_name", model_name)
            elif rtype == "step":
                # Mirror the producer's ``cycle_state.best_loss`` tracker: it is
                # updated by BOTH the pilot proxy (``record_cycle`` →
                # ``loss_valid``) AND the honest full-eval (``record_full_eval``
                # → ``loss_valid_full`` with ``loss_valid=None``). A fallback
                # that reads only ``loss_valid`` would drop the honest §5.1/§5.2
                # full-eval signal on an interrupted run (footer absent /
                # ``best_valid_loss: null``) and report the proxy best —
                # misrepresenting what the footer's ``best_valid_loss`` would
                # have recorded. Take the running min across both fields.
                for field in ("loss_valid", "loss_valid_full"):
                    raw = record.get(field)
                    if raw is None:
                        continue
                    candidate = _finite_loss_or_none(raw)
                    if candidate is None:
                        # Non-finite (NaN/inf from a diverged eval step) is not a
                        # measurement — skip it like a missing field so a glitched
                        # step never silently deposits NaN into the bootstrap.
                        continue
                    if min_step_loss is None or candidate < min_step_loss:
                        min_step_loss = candidate
                        min_step = record.get("step")
            elif rtype == "run_footer":
                # Arm identity lives on the footer regardless of whether the run
                # finished cleanly, so read it before the ``best_valid_loss`` null
                # guard below (an interrupted Tier-2 run still has a verifiable
                # arm). The block is emitted only when a progressive-freeze
                # controller drove the run (train_tg_lora L4603).
                pf_block = (record.get("tg_lora_summary") or {}).get(
                    "progressive_freeze"
                )
                if isinstance(pf_block, dict):
                    arm_policy = pf_block.get("policy")
                    arm_resolved_policy = pf_block.get("resolved_policy")
                    surrogate_seed = pf_block.get("surrogate_seed")
                    freeze_mode = pf_block.get("mode")
                # An interrupted run writes a footer with ``best_valid_loss: null``
                # (best_loss never updated) — treat that as "no footer value" and
                # fall through to the per-step minimum rather than crashing on
                # ``float(None)``. A NON-FINITE footer (``nan`` / ``inf`` — a
                # diverged run whose best_loss tracker went non-finite) is treated
                # the same way: it is not a citable arm result, so it falls through
                # to the per-step min rather than silently depositing NaN (see
                # :func:`_finite_loss_or_none`).
                raw_best = record.get("best_valid_loss")
                if raw_best is not None:
                    footer_value = _finite_loss_or_none(raw_best)
                    if footer_value is not None:
                        footer_step = record.get("best_valid_step")

    if footer_value is not None:
        value, best_step, source_kind = footer_value, footer_step, "run_footer"
    elif min_step_loss is not None:
        value, best_step, source_kind = min_step_loss, min_step, "min_loss_valid_step"
    else:
        raise ValueError(
            f"{path}: no best_valid_loss in run_footer and no step loss_valid lines"
        )

    provenance: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "model_name": model_name,
        "best_valid_step": best_step,
        "best_valid_loss_source": source_kind,
        "source": str(path),
        "arm_policy": arm_policy,
        "arm_resolved_policy": arm_resolved_policy,
        "surrogate_seed": surrogate_seed,
        "freeze_mode": freeze_mode,
    }
    return value, provenance


def _arm_label(value: float, prov: dict[str, Any]) -> str:
    seed = prov.get("seed")
    step = prov.get("best_valid_step")
    base = f"{prov['run_id']}(seed={seed},best_valid={value},step={step})"
    policy = prov.get("arm_policy")
    # Append the requested arm policy when the run carried progressive-freeze
    # provenance, so the deposit's per-arm note states output_first vs
    # random_order (the Tier-2 §4 order verdict's contrast) next to each float.
    if policy is not None:
        return f"{base},arm={policy}"
    return base


def _reject_swapped_arm(
    slot: str, provenances: Sequence[dict[str, Any]], *, must_be_surrogate: bool
) -> None:
    """Fail loud when a Tier-2 arm is deposited into the wrong slot.

    The Tier-2 §4 order verdict contrasts a real candidate arm
    (``output_first`` — ``surrogate_seed is None``) against a random-order
    surrogate (``surrogate_seed`` set). A candidate float deposited under
    ``--surrogate`` (or vice versa) flips the bootstrap sign of
    ``mean(surrogate) - mean(candidate)`` and silently turns SURPASSES into
    UNDERSHOOTS. Only multi-layer progressive arms (``mode == "progressive"``)
    carry the footer identity this checks; single-shot / no-progressive-freeze
    arms (the Tier-1 candidate and the Tier-1 baseline surrogate) are
    unverifiable and skipped.
    """
    for prov in provenances:
        if prov.get("freeze_mode") != "progressive":
            continue
        is_surrogate = prov.get("surrogate_seed") is not None
        if is_surrogate == must_be_surrogate:
            continue
        run_kind = "a random-order surrogate" if is_surrogate else "a real arm"
        raise ValueError(
            f"{prov.get('source', '<run>')}: this run is {run_kind} "
            f"(policy={prov.get('arm_policy')!r}, "
            f"surrogate_seed={prov.get('surrogate_seed')}) but was passed as "
            f"--{slot}. Swapping the candidate and surrogate arms inverts the "
            f"§4 verdict sign — re-check the --candidate / --surrogate "
            f"assignment."
        )


def form_deposit(
    candidate_paths: Sequence[str | Path],
    surrogate_paths: Sequence[str | Path],
    *,
    model: str,
    device: str,
    task: str = "generalize",
    architecture: str = "heterogeneous",
    base_seed: int = 0,
    total: int | None = None,
    seq_len: int | None = None,
    full_context: bool = False,
) -> dict[str, Any]:
    """Build a judge-ready §4 deposit from candidate + surrogate run artifacts.

    ``candidate`` is the output_first progressive-freeze arm; ``surrogate`` is
    the comparison arm (Tier-1 deposits the full-backprop baseline here per
    TASK-0152). One ``best_valid_loss`` per run becomes one entry in the
    corresponding losses array, in input order.

    ``seq_len`` / ``full_context`` tag the deposit's context: the sole config a
    12GB GPU fits is seq_len=256 (TASK-0152 lines 86-97), a reduced-context
    probe that is NOT the full §4 seq_len=1024 verdict. The honest default is
    reduced-context until a ``seq_len >= 1024`` (or explicit ``full_context``)
    proves otherwise — mis-labeling a reduced probe as the full verdict is the
    dangerous over-cite direction. The replay judge reads these fields:
    ``citable_as_target_scale`` opens for a real 9B run regardless of context,
    ``citable_as_full_section4_verdict`` only for a full-context deposit.
    """
    candidate_paths = [Path(p) for p in candidate_paths]
    surrogate_paths = [Path(p) for p in surrogate_paths]
    if not candidate_paths:
        raise ValueError("at least one --candidate run_metrics.jsonl is required")
    if not surrogate_paths:
        raise ValueError("at least one --surrogate run_metrics.jsonl is required")

    candidate_losses: list[float] = []
    candidate_labels: list[str] = []
    candidate_provenances: list[dict[str, Any]] = []
    for p in candidate_paths:
        value, prov = extract_best_valid_loss(p)
        candidate_losses.append(value)
        candidate_labels.append(_arm_label(value, prov))
        candidate_provenances.append(prov)

    surrogate_losses: list[float] = []
    surrogate_labels: list[str] = []
    surrogate_provenances: list[dict[str, Any]] = []
    for p in surrogate_paths:
        value, prov = extract_best_valid_loss(p)
        surrogate_losses.append(value)
        surrogate_labels.append(_arm_label(value, prov))
        surrogate_provenances.append(prov)

    # Tier-2 §4 order-verdict arm guard. c28e522 gave each progressive-freeze
    # run machine-readable arm identity in its footer (requested policy +
    # surrogate seed); this consumes it so a SWAPPED --candidate/--surrogate
    # — which silently inverts the verdict sign, the exact P0 hand-labeling
    # hazard the float-extraction removed — fails LOUD at form time. Only a
    # multi-layer progressive arm (mode=="progressive") carries verifiable
    # identity: a Tier-1 candidate (plain TG-LoRA, no progressive-freeze
    # footer) and a Tier-1 surrogate (full-backprop baseline) report no such
    # block, so the guard has nothing to check and stays silent there.
    _reject_swapped_arm("candidate", candidate_provenances, must_be_surrogate=False)
    _reject_swapped_arm("surrogate", surrogate_provenances, must_be_surrogate=True)

    arms_note = (
        f"candidate arms: {', '.join(candidate_labels)} | "
        f"surrogate(baseline) arms: {', '.join(surrogate_labels)}"
    )
    # full_context: seq_len>=1024 (or explicit --full-context) marks the deposit
    # the full §4 verdict; else reduced-context (the honest 12GB default). See
    # the docstring for the over-cite hazard.
    is_full_context = bool(
        full_context or (seq_len is not None and seq_len >= 1024)
    )
    return {
        "candidate_losses": candidate_losses,
        "surrogate_losses": surrogate_losses,
        "n_candidate": len(candidate_losses),
        "n_surrogate": len(surrogate_losses),
        # Per-arm requested policy surfaced from each footer's progressive-freeze
        # block (None where the run carried none — the Tier-1 candidate / the
        # Tier-1 baseline). Machine-readable arm identity end-to-end: footer →
        # deposit → (replay ignores unknown fields, so this is purely additive).
        "candidate_arm_policies": [p.get("arm_policy") for p in candidate_provenances],
        "surrogate_arm_policies": [p.get("arm_policy") for p in surrogate_provenances],
        # genuine target-scale recording — NOT proxy plumbing, synthetic, or a
        # negative control, so the §4 target-scale citation gate opens for it.
        "proxy_scale": False,
        "synthetic": False,
        "negative_control": False,
        # reduced-context provenance. ``citable_as_target_scale`` opens (it IS a
        # real 9B run); ``citable_as_full_section4_verdict`` only when full_context.
        "full_context": is_full_context,
        "seq_len": seq_len,
        "model": model,
        "device": device,
        "task": task,
        "architecture": architecture,
        "base_seed": base_seed,
        # symmetric per-arm budget => not a degraded-arm negative control.
        "total": total,
        "candidate_total": total,
        "surrogate_total": total,
        "source": (
            "Formed by scripts/form_freeze_validloss_deposit.py from upstream "
            "run_metrics.jsonl best_valid_loss (one float per seed per arm). "
            + arms_note
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="form_freeze_validloss_deposit",
        description=(
            "Form a GOAL §4 verdict-gate deposit from real upstream "
            "run_metrics.jsonl (candidate vs surrogate/baseline, multi-seed). "
            "Emits the schema scripts.replay_freeze_validloss_ci judges."
        ),
    )
    p.add_argument(
        "--candidate",
        nargs="+",
        required=True,
        metavar="RUN_METRICS_JSONL",
        help="one or more candidate (output_first progressive-freeze) run_metrics.jsonl",
    )
    p.add_argument(
        "--surrogate",
        nargs="+",
        required=True,
        metavar="RUN_METRICS_JSONL",
        help="one or more surrogate (Tier-1: full-backprop baseline) run_metrics.jsonl",
    )
    p.add_argument("--model", required=True, help="model name (e.g. Qwen/Qwen3.5-9B)")
    p.add_argument("--device", required=True, help="device label (e.g. cuda-rtx3060)")
    p.add_argument(
        "--task", default="generalize", help="verdict task (default generalize)"
    )
    p.add_argument(
        "--architecture",
        default="heterogeneous",
        help="verdict architecture (default heterogeneous)",
    )
    p.add_argument(
        "--base-seed", type=int, default=0, help="bootstrap base seed (default 0)"
    )
    p.add_argument(
        "--total",
        type=int,
        default=None,
        help="per-arm training budget (cycles); symmetric => not a negative control",
    )
    p.add_argument(
        "--seq-len", type=int, default=None,
        help="max seq_len the runs trained at; >=1024 marks the deposit the full "
             "§4 verdict (default None = reduced-context, the honest 12GB assumption).",
    )
    p.add_argument(
        "--full-context", action="store_true",
        help="mark the deposit the full §4 seq_len=1024 verdict; set ONLY for runs "
             "that fit full context (>12GB GPU), never a 12GB seq_len=256 probe.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="write the deposit JSON here; omit to print to stdout",
    )
    p.add_argument("--indent", type=int, default=2, help="JSON indent (default 2)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    deposit = form_deposit(
        args.candidate,
        args.surrogate,
        model=args.model,
        device=args.device,
        task=args.task,
        architecture=args.architecture,
        base_seed=args.base_seed,
        total=args.total,
        seq_len=args.seq_len,
        full_context=args.full_context,
    )
    text = json.dumps(deposit, indent=args.indent)
    if args.output:
        Path(args.output).write_text(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
