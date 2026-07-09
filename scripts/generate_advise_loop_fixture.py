#!/usr/bin/env python
"""Emit a ``run_metrics.jsonl`` via the REAL ``RunMetrics`` producer.

This is the producer-of-record for the committed fixture
``tests/fixtures/advise_loop/run_metrics_real_producer.jsonl``. It exists so the
dormant producer->consumer loop (RunMetrics producer -> ``advise_training.py``
consumer) is proven on **real producer code**, not a hand-built dict that merely
mirrors the schema: every byte of the fixture is written by
``src.utils.run_metrics.RunMetrics.record_step`` / ``write_header``. If the
producer ever renames a field the consumer reads (e.g. ``tg_lora_loss_pilot_eval``),
regenerating this fixture + the consumer tests go RED together.

The loss *values* are a synthetic plateau trajectory (improve 7 cycles, then
freeze). The genuine 9B run is Category-C on this mirror (private ``src.data`` +
>12 GB GPU, GOAL sec4), so a real-model trajectory is not producible here; the
SERIALIZATION / field contract — the part that was previously only
synthetically asserted — is now real.

Usage::

    python scripts/generate_advise_loop_fixture.py
    python scripts/generate_advise_loop_fixture.py --out /path/to/run_metrics.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# Standalone-CLI bootstrap: a bare ``python scripts/...`` invocation puts
# ``scripts/`` (not the repo root) on sys.path, so make the repo root importable
# so ``src.*`` resolves without a PYTHONPATH wrapper.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.run_metrics import RunMetrics  # noqa: E402


class _FakeCfg:
    """Minimal config object satisfying ``RunMetrics.write_header`` accessors.

    Mirrors the ``FakeCfg`` idiom in ``tests/test_run_metrics.py``. Only the
    attributes ``write_header`` reads are populated; everything else the header
    records is fetched via ``_cfg_value`` defaults.
    """

    class model:
        name_or_path = "Qwen/Qwen-real-producer-fixture"

    class training:
        batch_size = 1
        grad_accumulation = 1
        learning_rate = 1e-4

    class lora:
        r = 8
        alpha = 16

    class experiment:
        seed = 42

    _path = "tests/fixtures/advise_loop/run_metrics_real_producer.yaml"
    tg_lora = None
    alpha_line = None


def _plateau_steps(n_cycles: int = 14) -> list[dict[str, Any]]:
    """A plateau trajectory: loss improves for 7 cycles then goes flat.

    Flat tail -> stagnation + convergence -> the advisor fires ``stop_training``
    and ``increase_k`` (whose remediation names ``tg_lora.K_initial``), which is
    exactly the rendered advisory the consumer tests assert on.
    """
    steps: list[dict[str, Any]] = []
    for i in range(n_cycles):
        loss = round(2.0 - 0.10 * min(i, 6), 4)
        steps.append(
            {
                "step": i + 1,
                "cycle": i,
                "total_backward_passes": i + 1,
                "loss_train": loss,
                "loss_valid": loss,
                "grad_norm": 0.5,
                "tg_lora_accepted": True,
                "tg_lora_K": 3,
                "tg_lora_N": 2,
                "tg_lora_alpha": 0.5,
                # Real producer keys — the consumer must read these (NOT
                # loss_pilot / loss_after). Non-zero so a contract break shows
                # up as 0.0 in the consumer.
                "tg_lora_loss_pilot_eval": round(loss + 0.01, 4),
                "tg_lora_loss_after": round(loss - 0.005, 4),
            }
        )
    return steps


def emit(run_dir: str | Path, *, n_cycles: int = 14) -> Path:
    """Drive the REAL RunMetrics producer over a plateau trajectory.

    Returns the path to the emitted ``run_metrics.jsonl``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = RunMetrics(run_dir, mode="tg_lora", run_id="real_producer_fixture")
    metrics.write_header(_FakeCfg(), budget_type="cycles", budget_value=n_cycles)
    for kw in _plateau_steps(n_cycles):
        metrics.record_step(**kw)
    metrics.close()
    return run_dir / "run_metrics.jsonl"


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=str(repo / "tests" / "fixtures" / "advise_loop" / "run_metrics_real_producer.jsonl"),
        help="Output run_metrics.jsonl path (parent dir is created).",
    )
    parser.add_argument("--cycles", type=int, default=14, help="Number of plateau cycles.")
    args = parser.parse_args()

    out = Path(args.out)
    # Emit into the output dir so run_metrics.jsonl lands exactly at --out.
    emitted = emit(out.parent, n_cycles=args.cycles)
    if emitted != out:
        emitted.replace(out)
    print(f"REAL RunMetrics producer emitted {sum(1 for _ in out.open())} records -> {out}")


if __name__ == "__main__":
    main()
