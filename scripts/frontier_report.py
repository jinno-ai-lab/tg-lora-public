#!/usr/bin/env python
"""Generate frontier_report.json from Stage 3 memory frontier sweep results.

Reads per-seq-len aggregate_summary.json files produced by ``make paper-memory``
and classifies each run as completed / oom / failed for both baseline and TG-LoRA.
Produces a consolidated frontier_report.json with frontier separation detection.

For each run directory, also reads ``make_exit_code`` and ``make_output.log``
(written by ``run_frontier_sweep.sh``) so that OOM-from-log detection and
non-zero exit codes are reflected in the report — not silently defaulted.

Usage::

    python scripts/frontier_report.py \\
        --runs 1024:runs/s1024 2048:runs/s2048 3072:runs/s3072 \\
        --output runs/frontier_sweep/frontier_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OOM_PATTERNS = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"CUDA error", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    # The trainers' graceful-OOM handler logs "GPU out of memory (OOM) …" / the
    # baseline logs "OOM checkpoint saved to …". Recognize the bare acronym too
    # so a *handled* OOM (fault checkpoint saved, deferrable) is not misread as
    # a generic "failed" run — without this, only a kernel OOM-kill (exit 137) or
    # the literal "out of memory" substring classifies as OOM.
    re.compile(r"\bOOM\b"),
    re.compile(r"\bKilled\b"),
]


# Exit code the trainers emit for a *deferrable* GPU OOM (fault checkpoint saved,
# safe to retry at reduced batch). Kept as a local literal so this script stays
# stdlib-only; pinned equal to ``src.utils.device.OOM_EXIT_CODE`` by
# ``tests/test_fault_exit_contract.py`` so the two cannot drift.
OOM_EXIT_CODE = 3


def detect_oom_from_log(log: str) -> bool:
    return any(p.search(log) for p in OOM_PATTERNS)


def determine_status(
    *,
    exit_code: int,
    log: str,
    summary_exists: bool,
) -> str:
    if exit_code == 0 and summary_exists:
        return "completed"
    if exit_code == 137:
        return "oom"
    # A trainer-emitted deferrable-OOM exit code (graceful handler caught the
    # OOM and saved a fault checkpoint). Distinct from the kernel OOM-kill (137)
    # above and from a generic fault exit — recognize it before log scraping so
    # the verdict holds even when the log line was rotated/truncated.
    if exit_code == OOM_EXIT_CODE:
        return "oom"
    if detect_oom_from_log(log):
        return "oom"
    if exit_code != 0 and not summary_exists:
        return "failed"
    return "completed" if exit_code == 0 else "failed"


def _read_peak_mb(summary: dict[str, Any], key: str) -> float | None:
    agg = summary.get("aggregate", {}).get(key, {})
    mean = agg.get("mean")
    return float(mean) if isinstance(mean, (int, float)) else None


def find_frontier_boundary(runs: list[dict[str, Any]]) -> int | None:
    boundary = None
    for run in sorted(runs, key=lambda r: r["seq_len"]):
        if run["baseline_status"] != "completed" and run["tg_status"] == "completed":
            boundary = run["seq_len"]
    return boundary


def build_frontier_report(run_infos: list[dict[str, Any]]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []

    for info in run_infos:
        seq_len = info["seq_len"]
        run_dir = Path(info["run_dir"])
        summary_path = run_dir / "aggregate_summary.json"

        summary_exists = summary_path.exists()
        summary: dict[str, Any] = {}
        if summary_exists:
            summary = json.loads(summary_path.read_text())

        baseline_status = determine_status(
            exit_code=info["baseline_exit"],
            log=info.get("baseline_log", ""),
            summary_exists=summary_exists,
        )
        tg_status = determine_status(
            exit_code=info["tg_exit"],
            log=info.get("tg_log", ""),
            summary_exists=summary_exists,
        )

        frontier_separation = (
            baseline_status != "completed" and tg_status == "completed"
        )

        entry: dict[str, Any] = {
            "seq_len": seq_len,
            "run_dir": str(run_dir),
            "baseline_status": baseline_status,
            "tg_status": tg_status,
            "frontier_separation": frontier_separation,
        }

        if summary_exists:
            tg_peak = _read_peak_mb(summary, "warm_tg_gpu_peak_mb")
            bl_peak = _read_peak_mb(summary, "warm_baseline_gpu_peak_mb")
            entry["tg_peak_mb"] = tg_peak
            entry["baseline_peak_mb"] = bl_peak
            entry["summary_path"] = str(summary_path)
            if tg_peak is not None and bl_peak is not None:
                entry["memory_delta_mb"] = bl_peak - tg_peak
                if bl_peak > 0:
                    entry["memory_savings_pct"] = (bl_peak - tg_peak) / bl_peak * 100
        else:
            entry["summary_path"] = None

        runs.append(entry)

    boundary = find_frontier_boundary(runs)

    # Aggregate memory savings across completed runs
    deltas = [
        (r["baseline_peak_mb"], r["memory_delta_mb"])
        for r in runs
        if r.get("memory_delta_mb") is not None
    ]
    avg_savings_pct = None
    if deltas:
        total_bl = sum(bl for bl, _ in deltas)
        total_delta = sum(d for _, d in deltas)
        avg_savings_pct = total_delta / total_bl * 100 if total_bl > 0 else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seq_lens": [r["seq_len"] for r in runs],
        "runs": runs,
        "frontier_boundary": boundary,
        "frontier_separation_detected": boundary is not None,
        "avg_memory_savings_pct": avg_savings_pct,
    }


def _read_legacy_files(run_dir: Path, meta: dict[str, Any]) -> None:
    """Read make_exit_code and check aggregate_summary.json for legacy mode."""
    exit_code_path = run_dir / "make_exit_code"
    if exit_code_path.exists():
        try:
            meta["make_exit"] = int(exit_code_path.read_text().strip())
        except (ValueError, OSError):
            meta.setdefault("make_exit", 0)
    else:
        meta.setdefault("make_exit", 0)
    meta.setdefault("summary_exists", (run_dir / "aggregate_summary.json").exists())
    meta.setdefault("oom_in_log", False)


def _read_run_meta(run_dir: Path) -> dict[str, Any]:
    """Read per-run metadata written by run_frontier_sweep.sh.

    Primary source: ``run_metadata.json`` (structured JSON with make_exit,
    summary_exists, oom_in_log).  Falls back to reading ``make_exit_code``
    and ``make_output.log`` individually when the JSON file is absent.
    """
    meta: dict[str, Any] = {}

    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            raw = json.loads(metadata_path.read_text())
            # Handle frontier sweep format (has "seeds" array)
            if "seeds" in raw:
                seeds = raw["seeds"]
                # Any seed with baseline failure => baseline failed
                bl_failed = any(s.get("baseline_exit", 0) != 0 for s in seeds)
                tg_failed = any(s.get("tg_exit", 0) != 0 for s in seeds)
                bl_oom = any(s.get("baseline_oom", False) for s in seeds)
                tg_oom = any(s.get("tg_oom", False) for s in seeds)
                meta["make_exit"] = 1 if bl_failed else 0
                meta["tg_exit_override"] = 1 if tg_failed else 0
                meta["summary_exists"] = bool(raw.get("summary_exists", False))
                meta["oom_in_log"] = bl_oom
                meta["tg_oom_in_log"] = tg_oom
                meta["frontier_sweep_format"] = True
            else:
                meta["make_exit"] = int(raw.get("make_exit", 0))
                meta["summary_exists"] = bool(raw.get("summary_exists", False))
                meta["oom_in_log"] = bool(raw.get("oom_in_log", False))
        except (json.JSONDecodeError, ValueError, OSError):
            _read_legacy_files(run_dir, meta)
    else:
        _read_legacy_files(run_dir, meta)

    log_path = run_dir / "make_output.log"
    if log_path.exists():
        try:
            meta["make_log"] = log_path.read_text()
        except OSError:
            meta["make_log"] = ""
    else:
        meta["make_log"] = ""

    return meta


def _split_oom_log(log: str) -> tuple[str, str]:
    """Split log content into baseline and TG OOM-relevant lines.

    Mirrors the grep-based distinction ``run_frontier_sweep.sh`` previously
    attempted inline: lines containing both "baseline" and an OOM pattern are
    attributed to baseline, likewise for "tg".  Lines with an OOM pattern but
    no clear baseline/tg indicator are attributed to both (conservative).
    """
    baseline_lines: list[str] = []
    tg_lines: list[str] = []
    for line in log.splitlines():
        if not any(p.search(line) for p in OOM_PATTERNS):
            continue
        lower = line.lower()
        if "baseline" in lower:
            baseline_lines.append(line)
        elif "tg" in lower:
            tg_lines.append(line)
        else:
            baseline_lines.append(line)
            tg_lines.append(line)
    return "\n".join(baseline_lines), "\n".join(tg_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frontier_report.json from sweep runs")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="SEQ_LEN:RUN_DIR pairs (e.g. 1024:runs/s1024 2048:runs/s2048)",
    )
    parser.add_argument("--output", "-o", required=True, help="Output frontier_report.json path")
    args = parser.parse_args()

    run_infos: list[dict[str, Any]] = []
    for token in args.runs:
        parts = token.split(":", 1)
        if len(parts) != 2:
            print(f"ERROR: expected SEQ_LEN:RUN_DIR but got '{token}'", file=sys.stderr)
            sys.exit(2)
        seq_len = int(parts[0])
        run_dir = Path(parts[1])

        meta = _read_run_meta(run_dir)
        make_exit = meta["make_exit"]
        baseline_log, tg_log = _split_oom_log(meta["make_log"])

        # Use summary_exists from metadata (explicit signal from the shell
        # script) when available, otherwise check the filesystem.
        summary_exists = meta.get(
            "summary_exists",
            (run_dir / "aggregate_summary.json").exists(),
        )

        # Frontier sweep format: explicit per-component exit codes
        if meta.get("frontier_sweep_format"):
            tg_exit = meta.get("tg_exit_override", 0)
            if meta.get("oom_in_log"):
                baseline_log = baseline_log or "CUDA out of memory"
            if meta.get("tg_oom_in_log"):
                tg_log = tg_log or "CUDA out of memory"
        else:
            # Derive per-component exit codes.  ``make paper-memory`` runs
            # baseline and TG as a single suite, so ``make_exit`` applies to
            # both.  However, if the summary exists and no TG-specific OOM
            # patterns were found in the log, the TG phase completed OK.
            tg_exit = 0 if summary_exists and not tg_log else make_exit

        run_infos.append({
            "seq_len": seq_len,
            "run_dir": str(run_dir),
            "baseline_exit": make_exit,
            "tg_exit": tg_exit,
            "baseline_log": baseline_log,
            "tg_log": tg_log,
        })

    report = build_frontier_report(run_infos)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    status = "FRONTIER DETECTED" if report["frontier_separation_detected"] else "NO FRONTIER"
    print(f"Frontier report written to {out}")
    print(f"Boundary: {report['frontier_boundary']} ({status})")
    for r in report["runs"]:
        fs = " [FRONTIER]" if r["frontier_separation"] else ""
        print(f"  seq_len={r['seq_len']}: BL={r['baseline_status']}, TG={r['tg_status']}{fs}")


if __name__ == "__main__":
    main()
