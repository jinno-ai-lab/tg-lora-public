#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import orjson


CONDITIONS = ("baseline", "fixed_n", "cosine_n")


def load_run_metrics(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    header: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    footer: dict[str, Any] = {}
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        record = orjson.loads(line)
        kind = record.get("type", "step")
        if kind == "run_header":
            header = record
        elif kind == "run_footer":
            footer = record
        else:
            records.append(record)
    return header, records, footer


def summarize_run(path: Path) -> dict[str, Any]:
    header, records, footer = load_run_metrics(path)
    reference = header.get("comparison_reference") or {}
    first_loss = (
        float(reference["loss"])
        if isinstance(reference.get("loss"), (int, float))
        else _first_numeric(records, "loss_valid")
        or _first_numeric(records, "loss_train")
    )
    best_loss = footer.get("best_valid_loss")
    wall_seconds = footer.get("total_wall_seconds")
    loss_red_per_wall_minute = None
    if (
        isinstance(first_loss, (int, float))
        and isinstance(best_loss, (int, float))
        and isinstance(wall_seconds, (int, float))
        and wall_seconds > 0
    ):
        loss_red_per_wall_minute = (first_loss - best_loss) / (wall_seconds / 60.0)

    step_records = [
        record for record in records if record.get("type", "step") == "step"
    ]
    n_values = [
        int(record["tg_lora_N"])
        for record in step_records
        if isinstance(record.get("tg_lora_N"), int)
    ]
    proposed_n_values = [
        int(record["tg_lora_proposed_N"])
        for record in step_records
        if isinstance(record.get("tg_lora_proposed_N"), int)
    ]
    accepted_values = [
        record.get("tg_lora_accepted")
        for record in step_records
        if record.get("tg_lora_accepted") is not None
    ]
    rollback_count = sum(
        1 for record in step_records if record.get("tg_lora_rollback_triggered") is True
    )
    post_eval_count = sum(
        1
        for record in step_records
        if record.get("tg_lora_post_extrapolation_eval") is True
    )
    post_eval_skipped_count = sum(
        1
        for record in step_records
        if record.get("tg_lora_post_extrapolation_eval_skipped") is True
    )
    validation_forwards = sum(
        int(record.get("tg_lora_validation_forwards") or 0) for record in step_records
    )
    pilot_validation_forwards = sum(
        int(record.get("tg_lora_pilot_validation_forwards") or 0)
        for record in step_records
    )
    post_validation_forwards = sum(
        int(record.get("tg_lora_post_validation_forwards") or 0)
        for record in step_records
    )
    skip_reasons = Counter(
        str(record.get("tg_lora_post_extrapolation_eval_skip_reason"))
        for record in step_records
        if record.get("tg_lora_post_extrapolation_eval_skipped") is True
    )
    consistency_values = [
        float(record["tg_lora_predicted_consistency"])
        for record in step_records
        if isinstance(record.get("tg_lora_predicted_consistency"), (int, float))
    ]
    consistency_by_n: dict[str, list[float]] = defaultdict(list)
    for record in step_records:
        n = record.get("tg_lora_N")
        c = record.get("tg_lora_predicted_consistency")
        if isinstance(n, int) and isinstance(c, (int, float)):
            consistency_by_n[str(n)].append(float(c))

    summary = footer.get("tg_lora_summary") or {}
    return {
        "path": str(path),
        "mode": header.get("mode"),
        "optimizer_lifecycle": header.get("optimizer_lifecycle"),
        "wall_seconds": wall_seconds,
        "best_valid_loss": best_loss,
        "final_train_loss": footer.get("final_train_loss"),
        "first_observed_loss": first_loss,
        "loss_red_per_wall_minute": loss_red_per_wall_minute,
        "total_backward_passes": _last_numeric(step_records, "total_backward_passes"),
        "final_reduction_rate": _last_numeric(step_records, "tg_lora_reduction_rate"),
        "n_distribution": dict(sorted(Counter(n_values).items())),
        "proposed_n_distribution": dict(sorted(Counter(proposed_n_values).items())),
        "accepted_count": sum(1 for value in accepted_values if value is True),
        "rejected_count": sum(1 for value in accepted_values if value is False),
        "rollback_count": rollback_count,
        "rollback_rate": rollback_count / len(accepted_values)
        if accepted_values
        else None,
        "post_extrapolation_eval_count": post_eval_count,
        "post_extrapolation_eval_skipped_count": post_eval_skipped_count,
        "post_extrapolation_eval_skip_reasons": dict(sorted(skip_reasons.items())),
        "validation_forwards": validation_forwards,
        "pilot_validation_forwards": pilot_validation_forwards,
        "post_validation_forwards": post_validation_forwards,
        "mean_predicted_consistency": _mean(consistency_values),
        "mean_predicted_consistency_by_n": {
            n: _mean(values) for n, values in sorted(consistency_by_n.items())
        },
        "tg_lora_summary": summary,
    }


def summarize_suite(root: Path) -> dict[str, Any]:
    rows = []
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.replace("seed_", ""))
        row: dict[str, Any] = {"seed": seed}
        for condition in CONDITIONS:
            path = seed_dir / condition / "run_metrics.jsonl"
            if path.exists():
                row[condition] = summarize_run(path)
            else:
                row[condition] = {"missing": True, "path": str(path)}
        _add_deltas(row)
        rows.append(row)

    aggregate = {
        "suite": "cosine_n_ablation",
        "root": str(root),
        "conditions": {
            "baseline": "persistent Adam baseline, no extrapolation",
            "fixed_n": "TG-LoRA fixed N, persistent Adam, fixed lr",
            "cosine_n": "TG-LoRA cosine-driven N, persistent Adam, fixed lr",
        },
        "seeds": [row["seed"] for row in rows],
        "per_seed": rows,
        "aggregate": _aggregate(rows),
    }
    return aggregate


def _add_deltas(row: dict[str, Any]) -> None:
    baseline = row.get("baseline", {})
    fixed = row.get("fixed_n", {})
    cosine = row.get("cosine_n", {})
    row["comparisons"] = {
        "fixed_vs_baseline": _compare(fixed, baseline),
        "cosine_vs_baseline": _compare(cosine, baseline),
        "cosine_vs_fixed": _compare(cosine, fixed),
    }


def _compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "wall_seconds_delta": _delta(left, right, "wall_seconds"),
        "wall_seconds_ratio": _ratio(left, right, "wall_seconds"),
        "best_valid_loss_delta": _delta(left, right, "best_valid_loss"),
        "loss_red_per_wall_minute_ratio": _ratio(
            left, right, "loss_red_per_wall_minute"
        ),
        "reduction_rate_delta": _delta(left, right, "final_reduction_rate"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in CONDITIONS:
        for key in (
            "wall_seconds",
            "best_valid_loss",
            "loss_red_per_wall_minute",
            "final_reduction_rate",
            "rollback_rate",
            "validation_forwards",
            "pilot_validation_forwards",
            "post_validation_forwards",
            "post_extrapolation_eval_count",
            "post_extrapolation_eval_skipped_count",
            "mean_predicted_consistency",
        ):
            values = [
                row[condition].get(key)
                for row in rows
                if isinstance(row.get(condition, {}).get(key), (int, float))
            ]
            out[f"{condition}_{key}"] = _series(values)
    for comparison in ("fixed_vs_baseline", "cosine_vs_baseline", "cosine_vs_fixed"):
        for key in (
            "wall_seconds_ratio",
            "best_valid_loss_delta",
            "loss_red_per_wall_minute_ratio",
            "reduction_rate_delta",
        ):
            values = [
                row["comparisons"][comparison].get(key)
                for row in rows
                if isinstance(
                    row.get("comparisons", {}).get(comparison, {}).get(key),
                    (int, float),
                )
            ]
            out[f"{comparison}_{key}"] = _series(values)
    return out


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cosine-N Runtime Ablation",
        "",
        f"- Root: `{report['root']}`",
        f"- Seeds: {report['seeds']}",
        "",
        "## Per-Seed",
        "",
        "| Seed | Condition | Wall s | Best loss | Reduction rate | N distribution | Rollback rate | Val forwards | Post evals | Post skips | Mean consistency |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["per_seed"]:
        for condition in CONDITIONS:
            item = row[condition]
            lines.append(
                f"| {row['seed']} | {condition} | "
                f"{_fmt(item.get('wall_seconds'))} | "
                f"{_fmt(item.get('best_valid_loss'))} | "
                f"{_fmt(item.get('final_reduction_rate'))} | "
                f"`{item.get('n_distribution', {})}` | "
                f"{_fmt(item.get('rollback_rate'))} | "
                f"{_fmt(item.get('validation_forwards'), precision=0)} | "
                f"{_fmt(item.get('post_extrapolation_eval_count'), precision=0)} | "
                f"{_fmt(item.get('post_extrapolation_eval_skipped_count'), precision=0)} | "
                f"{_fmt(item.get('mean_predicted_consistency'))} |"
            )
    lines.extend(["", "## Aggregate", ""])
    for key, value in report["aggregate"].items():
        if value["mean"] is not None:
            lines.append(
                f"- `{key}`: mean={value['mean']:.4f}, values={value['values']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_numeric(records: list[dict[str, Any]], key: str) -> float | None:
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _last_numeric(records: list[dict[str, Any]], key: str) -> float | None:
    for record in reversed(records):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    a = left.get(key)
    b = right.get(key)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) - float(b)
    return None


def _ratio(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    a = left.get(key)
    b = right.get(key)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b != 0:
        if key == "loss_red_per_wall_minute" and b <= 0:
            return None
        return float(a) / float(b)
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _series(values: list[float]) -> dict[str, Any]:
    return {"values": values, "mean": _mean(values)}


def _fmt(value: Any, *, precision: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{value:.{precision}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    report = summarize_suite(args.root)
    json_path = args.root / "cosine_n_ablation_summary.json"
    md_path = args.root / "cosine_n_ablation_summary.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_markdown(report, md_path)
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))
    print(f"summary_json={json_path}")
    print(f"summary_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
