#!/usr/bin/env python
"""Replay a recorded order-sensitivity decomposition — no GPU, no model, no torch.

Companion to :mod:`scripts.replay_freeze_validloss_ci`. That script replayed the
GOAL §4 *verdict* (candidate vs surrogate) from recorded floats; this one
replays the *order-resolution diagnostic* — the
``Var(order)/Var(seed)`` variance decomposition that says whether the proxy
apparatus can resolve freeze order at all. The two are the two halves of the
proxy-scale evidence, and they deserve the same treatment:

* the verdict replay pinned the recorded ``TIES`` to the stored floats, so a
  GPU-less reader can confirm "no order advantage at proxy scale" is earned, not
  painted on;
* this replay pins the recorded ``ratio = 0.000`` — the result that converts
  *target-scale is assumed necessary* into *target-scale is proven necessary* —
  to the stored per-arm valid_loss samples, so the same reader can confirm the
  linchpin decomposition without re-running 24 training arms on a GPU.

The diagnostic (:mod:`scripts.run_freeze_order_sensitivity`) trains the
real progressive-freeze trio for ``n_orders`` distinct freeze orders at a fixed
seed and ``n_seeds`` seeds at a fixed order, then reports
``Var(order)`` / ``Var(seed)`` / their ``ratio``. All of that is a *measurement*
step; the *decomposition* itself is pure arithmetic over the recorded
``by_order`` / ``by_seed`` samples. This script re-runs only that arithmetic —
standard library only, no torch, no numpy, no model — so a committed recording
(e.g. ``tests/fixtures/freeze_order_sensitivity_proxy.json``, a real RTX 3060
run) is re-judged anywhere, and the recomputed ``ratio`` must match the one
recorded at run time. That pins the evidence is faithful rather than painted on:
the ratio is earned by the stored floats under the deterministic decomposition,
not asserted by the recording.

Target-scale drop-in works exactly as for the verdict replay: a real 9B run
deposits its own ``by_order`` / ``by_seed`` in the same schema, and replaying it
surfaces the target-scale ratio with **no code change**. ``proxy_scale`` is read
from the recording and surfaced, so a reader never cites the proxy ratio as a
target-scale result.

Usage::

    # Re-judge the committed proxy recording (the make target's default).
    make freeze-order-sensitivity-replay
    python -m scripts.replay_freeze_order_sensitivity \\
        tests/fixtures/freeze_order_sensitivity_proxy.json

    # Assert an expected resolution outcome; exit nonzero on mismatch (CI / gate).
    python -m scripts.replay_freeze_order_sensitivity samples.json --expected not_resolvable

    # The 9B target-scale drop-in: deposit by_order/by_seed in the same schema, then:
    python -m scripts.replay_freeze_order_sensitivity target_9b_order_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

# The two resolution outcomes the diagnostic's ratio maps to, surfaced as the
# ``--expected`` choices. ``ratio >= resolution_threshold`` (read from the
# recording) => ``resolvable``; otherwise ``not_resolvable``. Kept as plain
# strings (not imported from the torch-bearing diagnostic module) so this replay
# stays torch/numpy-free: a GPU-less verifier needs only the stdlib.
RESOLVABLE = "resolvable"
NOT_RESOLVABLE = "not_resolvable"
EXPECTED_OUTCOMES = (RESOLVABLE, NOT_RESOLVABLE)

# Fallback threshold if a recording omits ``resolution_threshold``. Matches the
# diagnostic's :data:`scripts.run_freeze_order_sensitivity.RESOLUTION_THRESHOLD`
# (0.10); the committed fixtures carry the field, so this only matters for
# hand-authored minimal recordings, and a torch-gated cross-check test pins the
# two equal.
DEFAULT_RESOLUTION_THRESHOLD = 0.10


def load_result(path: str | Path) -> dict[str, Any]:
    """Read a recorded order-sensitivity JSON and validate the decomposition schema.

    Accepts the object :func:`scripts.run_freeze_order_sensitivity.result_to_json`
    writes with ``--json --output`` (and any future target-scale run depositing
    ``by_order`` / ``by_seed`` in the same schema). The two sample lists the
    decomposition needs are required and must each have >= 2 entries (sample
    variance is undefined on fewer); every other field (``ratio``,
    ``resolution_threshold``, ``proxy_scale``, ``device``, ...) is optional
    provenance the report surfaces and cross-checks when present.
    """
    p = Path(path)
    with p.open() as fh:
        data = json.load(fh)
    for key in ("by_order", "by_seed"):
        value = data.get(key)
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError(
                f"{p}: '{key}' must be a list of >= 2 samples — not a recorded "
                f"order-sensitivity file (expected the schema from "
                f"`run_freeze_order_sensitivity --json`)"
            )
    return data


def _sample_variance(values: Sequence[float]) -> float:
    """Unbiased sample variance (n-1); 0.0 for fewer than two observations.

    Identical formula to :func:`scripts.run_freeze_order_sensitivity._variance`
    — reimplemented here (rather than imported) so this replay imports neither
    torch nor the diagnostic module: the decomposition is pure arithmetic a
    GPU-less verifier can run from the stdlib alone. A torch-gated test pins the
    two implementations equal on shared inputs so the local copy cannot drift.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def replay_order_sensitivity(data: dict[str, Any]) -> dict[str, Any]:
    """Re-derive the variance decomposition from stored per-arm samples.

    Deterministic and GPU-free: standard-library sample variance over the stored
    ``by_order`` / ``by_seed`` floats. The ``resolution_threshold`` is read from
    the recording (falling back to :data:`DEFAULT_RESOLUTION_THRESHOLD`) so the
    ``resolvable`` flag uses the threshold the recording was judged under, not a
    parallel constant — the replayed outcome then matches the recorded one for a
    consistent file (the faithfulness check in :func:`format_replay`).
    """
    threshold = float(data.get("resolution_threshold", DEFAULT_RESOLUTION_THRESHOLD))
    var_order = _sample_variance(data["by_order"])
    var_seed = _sample_variance(data["by_seed"])
    if var_seed > 0.0:
        ratio = var_order / var_seed
    else:
        # Mirror the diagnostic: a flat seed floor with moving order is +inf
        # (order visible on a dead floor), not a tidy 0; both-still is 0.
        ratio = 0.0 if var_order == 0.0 else float("inf")
    return {
        "var_order": var_order,
        "var_seed": var_seed,
        "ratio": ratio,
        "resolvable": ratio >= threshold,
        "resolution_threshold": threshold,
    }


def _outcome(resolvable: bool) -> str:
    return RESOLVABLE if resolvable else NOT_RESOLVABLE


def format_replay(path: str | Path, data: dict[str, Any], replay: dict[str, Any]) -> str:
    """Human-readable replay block: scale, the decomposition, and faithfulness.

    Faithfulness compares the replayed ``ratio`` and ``resolvable`` outcome to
    what the recording stored at run time: a match is the proof the stored floats
    earn the ratio under the deterministic decomposition, a mismatch is a warning
    that the file was edited inconsistently. The scale line makes
    ``proxy_scale`` visible so a reader never cites the proxy ratio as
    target-scale (or vice versa).
    """
    proxy_scale = bool(data.get("proxy_scale", True))
    scale = "PROXY" if proxy_scale else "TARGET"
    recorded_ratio = data.get("ratio")
    recorded_resolvable = data.get("resolvable")
    lines = [
        "freeze_order_sensitivity_replay — apparatus order-resolution "
        "on recorded samples (no GPU)",
        f"  source: {path}",
        f"  scale: {scale}_SCALE  (proxy_scale={proxy_scale})  "
        f"task={data.get('task', '?')}  architecture={data.get('architecture', '?')}",
        "",
        "  variance decomposition (recomputed from stored floats):",
        f"    Var(order) = {replay['var_order']:.12f}  "
        f"(spread across {len(data['by_order'])} orders at a fixed seed)",
        f"    Var(seed)  = {replay['var_seed']:.12f}  "
        f"(spread across {len(data['by_seed'])} seeds at a fixed order)",
        f"    ratio      = {replay['ratio']:.12f}  = Var(order)/Var(seed)",
        "",
        f"  verdict: order {_outcome(replay['resolvable']).upper()} "
        f"(ratio {'>=' if replay['resolvable'] else '<'} threshold "
        f"{replay['resolution_threshold']:.2f})",
    ]
    if recorded_ratio is not None:
        if _close(recorded_ratio, replay["ratio"]):
            lines.append(
                f"  faithfulness: replayed ratio MATCHES recording "
                f"({replay['ratio']:.12f})"
            )
        else:
            lines.append(
                f"  faithfulness: WARNING replayed ratio {replay['ratio']:.12f} "
                f"!= recorded {recorded_ratio:.12f}"
            )
    if recorded_resolvable is not None:
        if bool(recorded_resolvable) == replay["resolvable"]:
            lines.append(
                f"  faithfulness: replayed outcome MATCHES recording "
                f"({_outcome(replay['resolvable'])})"
            )
        else:
            lines.append(
                f"  faithfulness: WARNING replayed {_outcome(replay['resolvable'])} "
                f"!= recorded {_outcome(bool(recorded_resolvable))}"
            )
    if proxy_scale:
        lines.append(
            "  note: PROXY_SCALE — the ratio is from a 24-hidden proxy run, "
            "not the 9B target. The decomposition is faithful to the recorded "
            "run but is a proxy-scale result; do not cite it as target-scale. "
            "A genuine 9B run overwrites this file with real by_order/by_seed "
            "in the same schema and this conclusion upgrades."
        )
    else:
        lines.append(
            "  note: TARGET_SCALE — the recording is tagged target-scale "
            "(proxy_scale=False); this decomposition IS the target-scale "
            "order-resolution result. The proxy ratio upgrades to target-scale "
            "by swapping the sample source, with no code change."
        )
    return "\n".join(lines)


def _close(a: float, b: float) -> bool:
    """Float equality tolerant to the last ULP (both inf/0 handled)."""
    if a == b:
        return True
    if float("inf") in (abs(a), abs(b)):
        return a == b
    return abs(a - b) <= 1e-15 * max(1.0, abs(a), abs(b))


def replay_to_json(path: str | Path, data: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable replay: the recomputed decomposition plus provenance.

    ``citable_as_target_scale`` mirrors the verdict replay's gate: ``True`` only
    for a genuine target-scale recording (``proxy_scale=False``). The order
    diagnostic has no ``synthetic`` path of its own — a recording's floats are a
    measurement — so the gate is simply ``not proxy_scale`` here, and the verdict
    replay re-derives the stricter rule for its hand-authored plumbing fixtures.
    """
    return {
        "replayed_outcome": _outcome(replay["resolvable"]),
        "replayed_ratio": replay["ratio"],
        "replayed_var_order": replay["var_order"],
        "replayed_var_seed": replay["var_seed"],
        "recorded_ratio": data.get("ratio"),
        "recorded_outcome": (
            _outcome(bool(data["resolvable"])) if data.get("resolvable") is not None else None
        ),
        "faithful": (
            data.get("ratio") is None or _close(data["ratio"], replay["ratio"])
        ),
        "source": str(path),
        "proxy_scale": bool(data.get("proxy_scale", True)),
        "citable_as_target_scale": not bool(data.get("proxy_scale", True)),
        "resolution_threshold": replay["resolution_threshold"],
        "n_orders": len(data["by_order"]),
        "n_seeds": len(data["by_seed"]),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replay_freeze_order_sensitivity",
        description=(
            "Re-derive the recorded order-sensitivity variance decomposition "
            "(Var(order)/Var(seed)) from stored per-arm samples — no GPU, no "
            "model, no torch. Reads the JSON schema "
            "`run_freeze_order_sensitivity --json` writes (and the same schema a "
            "future 9B target run deposits), so a committed recording is "
            "verifiable anywhere and a target-scale sample file drops straight in."
        ),
    )
    p.add_argument(
        "samples_file",
        help="path to a recorded order-sensitivity JSON (from run_freeze_order_sensitivity --json).",
    )
    p.add_argument(
        "--expected", default=None, choices=EXPECTED_OUTCOMES,
        help="assert the replayed resolution outcome; exit nonzero (2) on "
             "mismatch — for CI / gate use that pins a recording to its expected "
             "outcome.",
    )
    p.add_argument("--json", action="store_true", help="emit the replay as JSON to stdout.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = load_result(args.samples_file)
    replay = replay_order_sensitivity(data)
    if args.json:
        print(json.dumps(replay_to_json(args.samples_file, data, replay), indent=2))
    else:
        print(format_replay(args.samples_file, data, replay))
    if args.expected is not None and _outcome(replay["resolvable"]) != args.expected:
        print(
            f"replay: EXPECTED {args.expected} but got "
            f"{_outcome(replay['resolvable'])}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
