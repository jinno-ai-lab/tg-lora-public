#!/usr/bin/env python
"""Summarize Component 2 landing runs and sidecar future-work diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.compare_runs import load_run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a paper-facing Component 2 summary from run_metrics.jsonl "
            "files. Main-result metrics and future-work sidecar diagnostics "
            "are intentionally separated."
        )
    )
    parser.add_argument("--baseline-runs", nargs="*", default=[])
    parser.add_argument("--proposal-runs", nargs="*", default=[])
    parser.add_argument("--future-runs", nargs="*", default=[])
    parser.add_argument(
        "--output-dir",
        default=f"runs/component2_landing_summary_{datetime.now():%Y%m%d_%H%M%S}",
    )
    return parser.parse_args()


def _as_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "run_metrics.jsonl"
    return p


def _finite(values: list[float | int | None]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p50": None, "p90": None}
    ordered = sorted(values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * p
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        w = pos - lo
        return ordered[lo] * (1 - w) + ordered[hi] * w

    return {"p10": pct(0.10), "p50": pct(0.50), "p90": pct(0.90)}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 1e-12 else None


def _theoretical_cost_ratio(m_steps: int | None) -> float | None:
    if m_steps is None or m_steps < 0:
        return None
    return 3.0 * (m_steps + 1) / (2.0 * m_steps + 3.0)


def _collect_alpha_losses(records: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for record in records:
        for key in ("alpha_line_exact_losses", "alpha_line_losses"):
            losses = record.get(key)
            if isinstance(losses, list):
                values.extend(_finite(losses))
                break
    return values


def _collect_future_pairs(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    pairs: list[dict[str, float]] = []
    for record in records:
        future = record.get("future_work")
        if not isinstance(future, dict):
            continue
        internal = future.get("internal")
        if not isinstance(internal, dict):
            continue
        raw_pairs = internal.get("g_dot_v_loss_delta_pairs")
        if not isinstance(raw_pairs, list):
            continue
        for pair in raw_pairs:
            if not isinstance(pair, dict):
                continue
            g_dot_v = pair.get("g_dot_v")
            loss_delta = pair.get("exact_loss_delta")
            if isinstance(g_dot_v, (int, float)) and isinstance(loss_delta, (int, float)):
                pairs.append(
                    {
                        "g_dot_v": float(g_dot_v),
                        "exact_loss_delta": float(loss_delta),
                    }
                )
    return pairs


def _collect_projection_ratios(
    footer: dict[str, Any] | None,
    records: list[dict[str, Any]],
) -> list[float]:
    values: list[float] = []
    summary = ((footer or {}).get("tg_lora_summary") or {}).get("future_work")
    if isinstance(summary, dict):
        values.extend(_finite(summary.get("projection_ratio_values") or []))
    for record in records:
        future = record.get("future_work")
        if not isinstance(future, dict):
            continue
        projection = future.get("projection_ratio")
        if not isinstance(projection, dict):
            continue
        value = projection.get("value")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def summarize_run(path: str) -> dict[str, Any]:
    metrics_path = _as_path(path)
    header, records, footer = load_run(metrics_path)
    summary = (footer or {}).get("tg_lora_summary") or {}
    total_wall_seconds = (footer or {}).get("total_wall_seconds")
    prefix_cache_build_seconds = summary.get("prefix_feature_cache_total_build_seconds")
    prefix_cache_load_seconds = summary.get("prefix_feature_cache_total_load_seconds")
    prefix_cache_seconds = sum(
        _finite([prefix_cache_build_seconds, prefix_cache_load_seconds])
    )
    wall_seconds_excluding_prefix_cache = (
        max(0.0, float(total_wall_seconds) - prefix_cache_seconds)
        if isinstance(total_wall_seconds, (int, float))
        else None
    )
    alpha_steps = _finite([r.get("alpha_line_alpha_steps") for r in records])
    alpha_losses = _collect_alpha_losses(records)
    train_losses = _finite([r.get("loss_train") for r in records])
    stop_reasons = Counter(
        r.get("alpha_line_stop_reason")
        for r in records
        if isinstance(r.get("alpha_line_stop_reason"), str)
    )
    m_steps = summary.get("alpha_line_m_alpha_steps", header.get("alpha_line_m_alpha_steps"))
    m_steps_int = int(m_steps) if isinstance(m_steps, int) else None
    future_pairs = _collect_future_pairs(records)
    pair_x = [p["g_dot_v"] for p in future_pairs]
    pair_y = [p["exact_loss_delta"] for p in future_pairs]
    projection_ratios = _collect_projection_ratios(footer, records)

    return {
        "path": str(metrics_path),
        "seed": header.get("seed"),
        "best_valid_loss": (footer or {}).get("best_valid_loss"),
        "total_wall_seconds": total_wall_seconds,
        "wall_seconds_excluding_prefix_cache": wall_seconds_excluding_prefix_cache,
        "prefix_cache_build_seconds": prefix_cache_build_seconds,
        "prefix_cache_load_seconds": prefix_cache_load_seconds,
        "prefix_cache_seconds": prefix_cache_seconds,
        "gpu_peak_mb": (footer or {}).get("gpu_peak_mb"),
        "total_backward_passes": (
            records[-1].get("total_backward_passes") if records else None
        ),
        "alpha_line_enabled": header.get("alpha_line_enabled"),
        "alpha_line_order": header.get("alpha_line_order"),
        "alpha_line_steps_total": summary.get("alpha_line_steps_total"),
        "alpha_line_steps_mean": _mean(alpha_steps),
        "alpha_line_stop_reasons": dict(stop_reasons),
        "alpha_line_v_update_wall_seconds_total": summary.get(
            "alpha_line_v_update_wall_seconds_total"
        ),
        "alpha_line_alpha_wall_seconds_total": summary.get(
            "alpha_line_alpha_wall_seconds_total"
        ),
        "theoretical_cost_ratio": _theoretical_cost_ratio(m_steps_int),
        "corpus_redundancy_proxy": {
            "paper_scope": "future_work_motivation",
            "source": "batch_loss_logs",
            "note": (
                "Uses logged train/alpha-line batch losses as a proxy. "
                "Per-sample logging is required for a true per-sample curve."
            ),
            "train_loss_quantiles": _quantiles(train_losses),
            "alpha_line_loss_quantiles": _quantiles(alpha_losses),
            "train_loss_count": len(train_losses),
            "alpha_line_loss_count": len(alpha_losses),
        },
        "future_work": {
            "paper_scope": "motivation_only",
            "projection_ratio_values": projection_ratios,
            "projection_ratio_mean": _mean(projection_ratios),
            "projection_ratio_count": len(projection_ratios),
            "internal": {
                "paper_exclude": True,
                "g_dot_v_loss_delta_pair_count": len(future_pairs),
                "g_dot_v_loss_delta_pearson": _pearson(pair_x, pair_y),
            },
        },
    }


def summarize_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    losses = _finite([r.get("best_valid_loss") for r in runs])
    walls = _finite([r.get("total_wall_seconds") for r in runs])
    walls_ex_cache = _finite([r.get("wall_seconds_excluding_prefix_cache") for r in runs])
    peaks = _finite([r.get("gpu_peak_mb") for r in runs])
    return {
        "count": len(runs),
        "best_valid_loss_mean": _mean(losses),
        "best_valid_loss_stdev": _stdev(losses),
        "wall_seconds_mean": _mean(walls),
        "wall_seconds_stdev": _stdev(walls),
        "wall_seconds_excluding_prefix_cache_mean": _mean(walls_ex_cache),
        "wall_seconds_excluding_prefix_cache_stdev": _stdev(walls_ex_cache),
        "gpu_peak_mb_mean": _mean(peaks),
        "gpu_peak_mb_stdev": _stdev(peaks),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.{digits}f}"
        return "nan"
    if value is None:
        return "n/a"
    return str(value)


def write_markdown(output: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Component 2 Landing Summary",
        "",
        "## Main Result Scope",
        "",
        "本文に使う対象は速度・メモリ・再現性です。future_work/internal は本文図表から除外します。",
        "",
        "| group | n | best valid mean | best valid sd | raw wall mean s | wall excl. prefix-cache s | peak VRAM mean MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name in ("baseline", "proposal"):
        group = payload["groups"][group_name]
        lines.append(
            "| "
            f"{group_name} | "
            f"{group['count']} | "
            f"{_fmt(group['best_valid_loss_mean'])} | "
            f"{_fmt(group['best_valid_loss_stdev'])} | "
            f"{_fmt(group['wall_seconds_mean'], 1)} | "
            f"{_fmt(group['wall_seconds_excluding_prefix_cache_mean'], 1)} | "
            f"{_fmt(group['gpu_peak_mb_mean'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Proposal Runs",
            "",
            "| seed | order | best valid | raw wall s | wall excl. prefix-cache s | prefix-cache s | peak MB | alpha steps | cost ratio | v wall s | alpha wall s | stop reasons |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for run in payload["runs"]["proposal"]:
        lines.append(
            "| "
            f"{_fmt(run['seed'], 0)} | "
            f"{_fmt(run['alpha_line_order'], 0)} | "
            f"{_fmt(run['best_valid_loss'])} | "
            f"{_fmt(run['total_wall_seconds'], 1)} | "
            f"{_fmt(run['wall_seconds_excluding_prefix_cache'], 1)} | "
            f"{_fmt(run['prefix_cache_seconds'], 1)} | "
            f"{_fmt(run['gpu_peak_mb'], 1)} | "
            f"{_fmt(run['alpha_line_steps_total'], 0)} | "
            f"{_fmt(run['theoretical_cost_ratio'])} | "
            f"{_fmt(run['alpha_line_v_update_wall_seconds_total'], 2)} | "
            f"{_fmt(run['alpha_line_alpha_wall_seconds_total'], 2)} | "
            f"{run['alpha_line_stop_reasons']} |"
        )

    future_runs = payload["runs"]["future"]
    if future_runs:
        lines.extend(
            [
                "",
                "## Future Work Sidecar",
                "",
                "論文末尾の動機づけ候補は projection ratio と corpus redundancy proxy のみです。"
                "internal correlation は JSON に保存しますが本文には出しません。",
                "",
                "| seed | projection ratio mean | projection ratio n | train loss p50 | alpha loss p50 | internal pairs |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in future_runs:
            future = run["future_work"]
            redundancy = run["corpus_redundancy_proxy"]
            lines.append(
                "| "
                f"{_fmt(run['seed'], 0)} | "
                f"{_fmt(future['projection_ratio_mean'])} | "
                f"{future['projection_ratio_count']} | "
                f"{_fmt(redundancy['train_loss_quantiles']['p50'])} | "
                f"{_fmt(redundancy['alpha_line_loss_quantiles']['p50'])} | "
                f"{future['internal']['g_dot_v_loss_delta_pair_count']} |"
            )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    baseline = [summarize_run(path) for path in args.baseline_runs]
    proposal = [summarize_run(path) for path in args.proposal_runs]
    future = [summarize_run(path) for path in args.future_runs]
    payload = {
        "created_at": datetime.now().isoformat(),
        "runs": {
            "baseline": baseline,
            "proposal": proposal,
            "future": future,
        },
        "groups": {
            "baseline": summarize_group(baseline),
            "proposal": summarize_group(proposal),
            "future": summarize_group(future),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "component2_landing_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(output_dir / "component2_landing_summary.md", payload)
    print(output_dir)


if __name__ == "__main__":
    main()
