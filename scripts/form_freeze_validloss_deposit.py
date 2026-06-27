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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def extract_best_valid_loss(path: str | Path) -> tuple[float, dict[str, Any]]:
    """Read the best valid loss + provenance from one ``run_metrics.jsonl``.

    Prefers the durable ``run_footer.best_valid_loss`` field; falls back to the
    minimum ``loss_valid`` across ``step`` lines for runs that predate the
    footer. Returns ``(value, provenance)`` where provenance names the run so a
    deposited float always traces back to its source artifact.
    """
    path = Path(path)
    run_id = path.stem
    seed: Any = None
    model_name: Any = None
    footer_value: float | None = None
    footer_step: Any = None
    min_step_loss: float | None = None
    min_step: Any = None
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
                loss_valid = record.get("loss_valid")
                if loss_valid is not None and (
                    min_step_loss is None or loss_valid < min_step_loss
                ):
                    min_step_loss = float(loss_valid)
                    min_step = record.get("step")
            elif rtype == "run_footer" and "best_valid_loss" in record:
                footer_value = float(record["best_valid_loss"])
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
    }
    return value, provenance


def _arm_label(value: float, prov: dict[str, Any]) -> str:
    seed = prov.get("seed")
    step = prov.get("best_valid_step")
    return f"{prov['run_id']}(seed={seed},best_valid={value},step={step})"


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
) -> dict[str, Any]:
    """Build a judge-ready §4 deposit from candidate + surrogate run artifacts.

    ``candidate`` is the output_first progressive-freeze arm; ``surrogate`` is
    the comparison arm (Tier-1 deposits the full-backprop baseline here per
    TASK-0152). One ``best_valid_loss`` per run becomes one entry in the
    corresponding losses array, in input order.
    """
    candidate_paths = [Path(p) for p in candidate_paths]
    surrogate_paths = [Path(p) for p in surrogate_paths]
    if not candidate_paths:
        raise ValueError("at least one --candidate run_metrics.jsonl is required")
    if not surrogate_paths:
        raise ValueError("at least one --surrogate run_metrics.jsonl is required")

    candidate_losses: list[float] = []
    candidate_labels: list[str] = []
    for p in candidate_paths:
        value, prov = extract_best_valid_loss(p)
        candidate_losses.append(value)
        candidate_labels.append(_arm_label(value, prov))

    surrogate_losses: list[float] = []
    surrogate_labels: list[str] = []
    for p in surrogate_paths:
        value, prov = extract_best_valid_loss(p)
        surrogate_losses.append(value)
        surrogate_labels.append(_arm_label(value, prov))

    arms_note = (
        f"candidate arms: {', '.join(candidate_labels)} | "
        f"surrogate(baseline) arms: {', '.join(surrogate_labels)}"
    )
    return {
        "candidate_losses": candidate_losses,
        "surrogate_losses": surrogate_losses,
        "n_candidate": len(candidate_losses),
        "n_surrogate": len(surrogate_losses),
        # genuine target-scale recording — NOT proxy plumbing, synthetic, or a
        # negative control, so the §4 citation gate opens for it.
        "proxy_scale": False,
        "synthetic": False,
        "negative_control": False,
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
