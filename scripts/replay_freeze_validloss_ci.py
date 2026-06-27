#!/usr/bin/env python
"""Replay recorded valid_loss samples through the GOAL §4 judge — no GPU, no model.

This is the Category-C (GPU) blocker **reduced to a concrete, GPU-free,
executable command** — the move the loop's feedback asked for in place of more
CPU scaffolding. Two roles:

1. **Verifiable recorded evidence.** ``scripts.run_freeze_validloss_ci`` trains
   the proxy and deposits *real* valid_loss samples (candidate + surrogate)
   plus its verdict into a JSON file (``--json --output``). A committed
   recording of that file — e.g. ``tests/fixtures/freeze_validloss_generalize_proxy.json``,
   a real RTX 3060 run — is evidence a reader can re-judge *anywhere* with no
   GPU and no model, and the recomputed verdict must match the one recorded at
   run time. That pins the evidence is faithful rather than painted on: the
   verdict is earned by the stored floats under the deterministic bootstrap,
   not asserted by the recording.

2. **The target-scale drop-in.** The 9B target run (private ``src.data``,
   Category-C) cannot run in this public mirror, but it deposits samples in the
   *same* schema. Dropping a ``proxy_scale: false`` sample file in and running
   this command yields the target-scale §4 verdict with **no code change** —
   the proxy label upgrades to target-scale purely by swapping the sample
   source. ``proxy_scale`` is read from the file and surfaced in the report, so
   a reader always sees which scale a replayed verdict is from.

   **The synthetic-provenance guard.** A recording may carry ``synthetic: true``
   to mark its floats as hand-authored plumbing (a constructed separation that
   exercises this branch), not a measurement. Such a recording is still judged —
   the verdict is faithfully recomputed from the stored floats — but it is never
   *presented* as a citable §4 result: the rendered note withholds the "this
   verdict IS the §4 target-scale result" claim a genuine 9B recording earns and
   instead says plainly "synthetic — do not cite". This converts the feedback's
   "every committed verdict is still proxy-scale and must not be cited as a §4
   target-scale result" warning from prose into a code-enforced contract, so a
   ``proxy_scale: false`` plumbing fixture can never be mistaken for a real 9B
   run. A genuine recording omits ``synthetic`` (or sets it ``false``).

   **The negative-control provenance guard.** A recording may carry
   ``negative_control: true`` to mark one of its arms as deliberately degraded
   on a non-order lever (an asymmetric training budget, ``candidate_total`` or
   ``surrogate_total``). Such a recording is a real measurement and is judged
   faithfully, but its verdict is an apparatus-sensitivity probe (proof the
   gate fires a genuine non-TIES label on a measured loss gap), never a §4
   order result — an additive note names the degraded arm and says so, and the
   ``citable_as_target_scale`` gate withholds the citation claim for it just as
   it does for a synthetic recording. Degrading the candidate fires the
   DOWNWARD label (UNDERSHOOTS); degrading the surrogate fires the UPWARD label
   (SURPASSES) — the symmetric completion, the only real-label direction that
   had never been recorded before this lever existed (the sole prior SURPASSES
   is the synthetic plumbing fixture).

   The same rule is also surfaced as a machine-readable
   ``citable_as_target_scale`` boolean in :func:`replay_to_json` (``True`` only
   for a genuine target-scale recording) so a downstream consumer does not have
   to infer citability from the raw ``proxy_scale`` / ``synthetic`` flags — the
   rendered claim and the machine gate enforce the same contract and cannot
   drift apart.

The replay re-runs *only* :func:`src.tg_lora.freeze_surrogate_ci.surrogate_valid_loss_ci`
— pure numpy over the stored floats, so it is deterministic and device-free.
It does not retrain, does not import torch, and does not assume a label: it
emits whatever the bootstrap CI on the stored samples says
(``SURPASSES`` / ``TIES`` / ``UNDERSHOOTS``).

Usage::

    # Re-judge the committed proxy recording (the make target's default).
    make freeze-replay
    python -m scripts.replay_freeze_validloss_ci \\
        tests/fixtures/freeze_validloss_generalize_proxy.json

    # Assert an expected verdict; exit nonzero on mismatch (CI / gate use).
    python -m scripts.replay_freeze_validloss_ci samples.json --expected TIES

    # The 9B target-scale drop-in: produce samples in the same schema, then:
    python -m scripts.replay_freeze_validloss_ci target_9b_samples.json

    # Override the materiality margin or the bootstrap RNG seed.
    python -m scripts.replay_freeze_validloss_ci samples.json \\
        --material-margin 0.05 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.tg_lora.freeze_surrogate_ci import (
    SurrogateValidLossCI,
    format_surrogate_valid_loss_ci,
    surrogate_valid_loss_ci,
)
from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES, UNDERSHOOTS

# The three verdicts the §4 judge can emit (imported so ``--expected`` choices
# are the exact labels the bootstrap layer returns, not a parallel vocabulary).
EXPECTED_VERDICTS = (SURPASSES, TIES, UNDERSHOOTS)


def load_samples(path: str | Path) -> dict[str, Any]:
    """Read a recorded-sample JSON and validate the §4 schema.

    Accepts the exact object :func:`scripts.run_freeze_validloss_ci.result_to_json`
    writes with ``--json --output`` (and any future target-scale run that
    deposits samples in the same schema). The two sample lists the judge needs
    are required and must be non-empty; every other field (``verdict``,
    ``base_seed``, ``proxy_scale``, ``synthetic``, ``task``, ...) is optional
    provenance the report surfaces when present.
    """
    p = Path(path)
    with p.open() as fh:
        data = json.load(fh)
    for key in ("candidate_losses", "surrogate_losses"):
        value = data.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"{p}: missing non-empty '{key}' — not a recorded-sample file "
                f"(expected the schema from `run_freeze_validloss_ci --json`)"
            )
    return data


def replay_samples(
    data: dict[str, Any],
    *,
    material_margin: float = 0.0,
    seed: int | None = None,
) -> SurrogateValidLossCI:
    """Re-run the §4 bootstrap judge on stored samples.

    Deterministic and GPU-free: the bootstrap is pure numpy over the stored
    floats. ``seed`` defaults to the file's recorded ``base_seed`` — the seed
    :func:`scripts.run_freeze_validloss_ci.run_ci` used for its own bootstrap —
    so a replay reproduces the recorded verdict bit-for-bit; pass an explicit
    ``seed`` to resample under a different RNG.
    """
    if seed is None:
        seed = int(data.get("base_seed", 0))
    return surrogate_valid_loss_ci(
        data["candidate_losses"],
        data["surrogate_losses"],
        seed=seed,
        material_margin=material_margin,
    )


def _resolve_negative_control_arm(data: dict[str, Any]) -> str | None:
    """Which arm a negative-control recording degraded, or ``None``.

    New recordings carry ``negative_control_arm`` directly. Legacy recordings
    (the committed UNDERSHOOTS fixture predates the field) are resolved by
    comparing each arm's recorded budget against ``total``: a budget that
    diverged was deliberately degraded. Returns ``'candidate'`` / ``'surrogate'``
    / ``'both'`` / ``None`` — so a surrogate-degraded recording (real SURPASSES)
    names the surrogate, not the candidate, keeping the provenance honest.
    """
    arm = data.get("negative_control_arm")
    if arm is not None:
        return arm
    total = data.get("total")
    if total is None:
        return None
    cand = data.get("candidate_total", total)
    surr = data.get("surrogate_total", total)
    cand_div = cand != total
    surr_div = surr != total
    if cand_div and surr_div:
        return "both"
    if cand_div:
        return "candidate"
    if surr_div:
        return "surrogate"
    return None


def format_replay(path: str | Path, data: dict[str, Any], ci: SurrogateValidLossCI) -> str:
    """Human-readable replay block: scale, the §4 verdict, and faithfulness.

    Faithfulness compares the replayed verdict to the ``verdict`` the recording
    stored at run time: a match is the proof the stored floats earn the verdict
    under the deterministic bootstrap, a mismatch is a warning that the file was
    edited inconsistently. The scale line makes ``proxy_scale`` visible so a
    reader never cites a proxy verdict as target-scale (or vice versa).

    ``synthetic`` (read from the recording, default ``False``) is the
    provenance guard: a hand-authored plumbing recording is judged but never
    presented as a citable §4 result — its note withholds the "this verdict IS
    the §4 target-scale result" claim a genuine recording earns, enforcing the
    feedback's "do not cite as a target-scale result" warning in the rendered
    output rather than relying on prose alone.

    ``negative_control`` (default ``False``) is the apparatus-provenance guard:
    a recording whose candidate arm was deliberately degraded on a non-order
    lever (an asymmetric training budget). Such a recording IS a real
    measurement and is judged faithfully, but its verdict is a sensitivity
    probe, not a §4 order result — an additive note says so plainly, keeping a
    recorded UNDERSHOOTS from being misread as an order disadvantage.
    """
    proxy_scale = bool(data.get("proxy_scale", True))
    synthetic = bool(data.get("synthetic", False))
    negative_control = bool(data.get("negative_control", False))
    scale = "PROXY" if proxy_scale else "TARGET"
    recorded = data.get("verdict")
    lines = [
        "freeze_replay — GOAL §4 judge on recorded samples (no GPU)",
        f"  source: {path}",
        f"  scale: {scale}_SCALE  (proxy_scale={proxy_scale}, "
        f"synthetic={synthetic}, negative_control={negative_control})  "
        f"task={data.get('task', '?')}  architecture={data.get('architecture', '?')}",
        "",
        format_surrogate_valid_loss_ci(ci),
    ]
    if recorded is not None:
        if ci.significance_verdict == recorded:
            lines.append(
                f"  faithfulness: replayed verdict MATCHES recording "
                f"({ci.significance_verdict})"
            )
        else:
            lines.append(
                f"  faithfulness: WARNING replayed {ci.significance_verdict} "
                f"!= recorded {recorded}"
            )
    # Scale-honesty note. ``synthetic`` takes precedence over both PROXY and
    # TARGET: a plumbing recording's floats are not a measurement, so the
    # verdict — though faithfully recomputed — is never citable as a §4 result
    # at any scale. A genuine recording reaches the PROXY/TARGET branches.
    if synthetic:
        lines.append(
            "  note: SYNTHETIC — the stored floats are hand-authored plumbing "
            f"(a constructed separation), not a {scale.lower()}-scale run. The "
            "verdict is faithfully recomputed from those floats but is "
            f"*synthetic* {scale.lower()}-scale evidence; do not cite it as a "
            "§4 result at any scale. A genuine run overwrites this file with "
            "real floats in the same schema and this note drops."
        )
    elif proxy_scale:
        lines.append(
            "  note: PROXY_SCALE — samples are from a 24-hidden proxy run, not "
            "the 9B target. The verdict is faithful to the recorded run but is "
            "a proxy-scale §4 result; do not cite it as target-scale."
        )
    else:
        # Target-scale. A genuine target-scale recording grants the citable
        # claim; a negative-control recording does NOT (its verdict is a
        # sensitivity probe, never a §4 order result), so the "this verdict IS
        # the §4 target-scale result" claim is withheld and the additive
        # NEGATIVE_CONTROL note below says why — keeping the prose claim and the
        # machine ``citable_as_target_scale`` gate from drifting apart.
        if negative_control:
            arm = _resolve_negative_control_arm(data)
            arm_clause = (
                "the candidate arm was" if arm != "surrogate"
                else "the surrogate arm was"
            )
            lines.append(
                "  note: TARGET_SCALE — samples are from a 9B run, so the "
                "recording IS at target scale; however a negative control was "
                f"applied ({arm_clause} deliberately degraded on a non-order "
                "lever), so the verdict is a sensitivity probe recorded at "
                "target scale, NOT a citable §4 order result."
            )
        else:
            lines.append(
                "  note: TARGET_SCALE — samples are from a 9B run; this verdict "
                "IS the §4 target-scale result. The proxy verdict upgrades to "
                "target-scale by swapping the sample source, with no code change."
            )
    # Negative-control provenance (additive — a negative control IS a real
    # measurement, just not of order). The degraded arm is named so the gap's
    # source is honest: a candidate-degraded recording's gap is from
    # undertraining the candidate (a recorded UNDERSHOOTS must not be misread
    # as "the output-first order is worse than random"), while a surrogate-
    # degraded recording's gap makes the candidate look better by construction
    # (a recorded SURPASSES that is a sensitivity probe, not an order win). The
    # verdict itself is recomputed from the stored floats; this note states only
    # why it is never a §4 order result, never which label it earned.
    if negative_control:
        arm = _resolve_negative_control_arm(data)
        if arm == "surrogate":
            arm_phrase = "the surrogate arm was deliberately degraded"
        elif arm == "both":
            arm_phrase = "both arms were deliberately degraded asymmetrically"
        else:  # "candidate" (and the legacy default for pre-arm recordings)
            arm_phrase = "the candidate arm was deliberately degraded"
        lines.append(
            f"  note: NEGATIVE_CONTROL — {arm_phrase} (an asymmetric training "
            "budget, unrelated to freeze order) to inject a real quality gap. "
            "The verdict is faithfully recomputed from the stored floats but is "
            "an apparatus-sensitivity probe, NOT a §4 order result; do not read "
            "it as evidence for or against an output-first order advantage."
        )
    return "\n".join(lines)


def replay_to_json(path: str | Path, data: dict[str, Any], ci: SurrogateValidLossCI) -> dict[str, Any]:
    """Machine-readable replay: the judge output plus the file's provenance.

    ``citable_as_target_scale`` is the single boolean a downstream consumer
    checks before citing a recording's verdict as a §4 target-scale result —
    ``True`` only for a genuine target-scale recording
    (``proxy_scale=False`` and not ``synthetic``). It mirrors the prose rule
    :func:`format_replay` renders, so the human-readable claim and the machine
    gate can never drift apart.
    """
    return {
        "replayed_verdict": ci.significance_verdict,
        "recorded_verdict": data.get("verdict"),
        "faithful": (
            data.get("verdict") is None
            or ci.significance_verdict == data.get("verdict")
        ),
        "source": str(path),
        "proxy_scale": bool(data.get("proxy_scale", True)),
        "synthetic": bool(data.get("synthetic", False)),
        "negative_control": bool(data.get("negative_control", False)),
        # Which arm a negative control degraded (resolved from
        # ``negative_control_arm`` when present, else inferred from the arms'
        # recorded budgets vs ``total``). Surfaced so a downstream consumer can
        # tell a candidate-degraded (DOWNWARD) from a surrogate-degraded
        # (UPWARD) sensitivity probe without re-deriving it from the budgets.
        "negative_control_arm": _resolve_negative_control_arm(data),
        # Machine-readable citation gate (GOAL §4): this recording's verdict MAY
        # be cited as a §4 target-scale result only when it is target-scale AND
        # genuine AND not a negative control — the exact prose rule
        # ``format_replay`` renders (a proxy, synthetic, or negative-control
        # recording withholds the "this verdict IS the §4 target-scale result"
        # claim). Surfacing it as one boolean closes the contract on the machine
        # path too, so a consumer does not infer citability from raw flags: this
        # is the feedback's "must not be cited as a §4 target-scale result"
        # warning enforced as a field, not prose.
        "citable_as_target_scale": (
            not bool(data.get("proxy_scale", True))
            and not bool(data.get("synthetic", False))
            and not bool(data.get("negative_control", False))
        ),
        "candidate_mean": ci.candidate_mean,
        "surrogate_mean": ci.surrogate_mean,
        "point_improvement": ci.point_improvement,
        "lower": ci.lower,
        "upper": ci.upper,
        "confidence": ci.confidence,
        "is_material": ci.is_material,
        "is_thin_evidence": ci.is_thin_evidence,
        "n_candidate": ci.n_candidate,
        "n_surrogate": ci.n_surrogate,
        "seed": ci.seed,
        "material_margin": ci.material_margin,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replay_freeze_validloss_ci",
        description=(
            "Re-judge recorded valid_loss samples through the GOAL §4 "
            "surrogate_valid_loss_ci judge — no GPU, no model. Reads the JSON "
            "schema `run_freeze_validloss_ci --json --output` writes (and the "
            "same schema a future 9B target run deposits), so a committed "
            "recording is verifiable anywhere and a target-scale sample file "
            "drops straight in."
        ),
    )
    p.add_argument(
        "samples_file",
        help="path to a recorded-sample JSON (from run_freeze_validloss_ci --json).",
    )
    p.add_argument(
        "--material-margin", type=float, default=0.0,
        help="minimum point_improvement for is_material (§7 significance vs "
             "materiality separation); default 0.0.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="bootstrap RNG seed; default is the file's recorded base_seed "
             "(reproduces the recorded verdict bit-for-bit).",
    )
    p.add_argument(
        "--expected", default=None, choices=EXPECTED_VERDICTS,
        help="assert the replayed verdict; exit nonzero (2) on mismatch — for "
             "CI / gate use that pins a recording to its expected outcome.",
    )
    p.add_argument("--json", action="store_true", help="emit the replay as JSON to stdout.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = load_samples(args.samples_file)
    ci = replay_samples(data, material_margin=args.material_margin, seed=args.seed)
    if args.json:
        print(json.dumps(replay_to_json(args.samples_file, data, ci), indent=2))
    else:
        print(format_replay(args.samples_file, data, ci))
    if args.expected is not None and ci.significance_verdict != args.expected:
        print(
            f"replay: EXPECTED {args.expected} but got {ci.significance_verdict}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
