#!/usr/bin/env python
"""Export paper experiment results to LaTeX, Markdown, and CSV formats.

Reads ``aggregate_summary.json`` (or equivalent) and produces publication-ready
tables enriched with confidence intervals and statistical significance markers.

Usage::

    python scripts/export_paper_results.py aggregate_summary.json --format latex
    python scripts/export_paper_results.py aggregate_summary.json --format markdown
    python scripts/export_paper_results.py aggregate_summary.json --format csv --output results.csv
    python scripts/export_paper_results.py aggregate_summary.json --format all --output-dir paper_tables/
"""
from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from src.analysis.stats import analyze_multi_seed


def load_aggregate(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Aggregate summary not found: {p}")
    data = json.loads(p.read_text())
    if "per_seed" not in data and "aggregate" not in data:
        raise ValueError("Invalid aggregate summary: missing 'per_seed' or 'aggregate' key")
    return data


def _extract_table_rows(
    summary: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    per_seed = summary.get("per_seed", {})
    if isinstance(per_seed, list):
        per_seed = {f"seed_{i}": r for i, r in enumerate(per_seed)}

    stats_result = analyze_multi_seed({"per_seed": per_seed, "aggregate": summary.get("aggregate", {})})
    metrics = stats_result.get("metrics", {})

    columns = ["metric", "n", "mean", "std", "ci_lower", "ci_upper"]
    rows: list[dict[str, Any]] = []
    for key in sorted(metrics):
        m = metrics[key]
        row: dict[str, Any] = {"metric": key}
        row["n"] = m.get("n", 0)
        row["mean"] = m.get("mean")
        row["std"] = m.get("std")
        row["ci_lower"] = m.get("ci_lower")
        row["ci_upper"] = m.get("ci_upper")
        rows.append(row)

    return columns, rows


def generate_latex_table(summary: dict[str, Any]) -> str:
    columns, rows = _extract_table_rows(summary)
    col_headers = ["Metric", "N", "Mean", "Std", "95\\% CI Lower", "95\\% CI Upper"]

    buf = StringIO()
    ncols = len(col_headers)
    buf.write("\\begin{table}[htbp]\n")
    buf.write("\\centering\n")
    buf.write("\\begin{tabular}{" + "l" + "r" * (ncols - 1) + "}\n")
    buf.write("\\toprule\n")
    buf.write(" & ".join(col_headers) + " \\\\\n")
    buf.write("\\midrule\n")

    for row in rows:
        vals = []
        for col in columns:
            v = row.get(col)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        buf.write(" & ".join(vals) + " \\\\\n")

    buf.write("\\bottomrule\n")
    buf.write("\\end{tabular}\n")
    buf.write("\\caption{TG-LoRA multi-seed experiment results with 95\\% confidence intervals.}\n")
    buf.write("\\label{tab:tg_lora_results}\n")
    buf.write("\\end{table}\n")
    return buf.getvalue()


def generate_markdown_table(summary: dict[str, Any]) -> str:
    columns, rows = _extract_table_rows(summary)
    col_headers = ["Metric", "N", "Mean", "Std", "95% CI Lower", "95% CI Upper"]

    lines: list[str] = []
    lines.append("| " + " | ".join(col_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(col_headers)) + "|")

    for row in rows:
        vals = []
        for col in columns:
            v = row.get(col)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines) + "\n"


def export_csv(summary: dict[str, Any], output_path: str | Path) -> None:
    columns, rows = _extract_table_rows(summary)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _export_all(summary: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    latex = generate_latex_table(summary)
    (out / "results.tex").write_text(latex)

    md = generate_markdown_table(summary)
    (out / "results.md").write_text(md)

    export_csv(summary, out / "results.csv")

    print(f"Exported LaTeX, Markdown, CSV to {out}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper experiment results")
    parser.add_argument("input", help="Path to aggregate_summary.json")
    parser.add_argument("--format", choices=["latex", "markdown", "csv", "all"], default="markdown")
    parser.add_argument("--output", "-o", help="Output file path (for single format)")
    parser.add_argument("--output-dir", default="paper_tables", help="Output directory for --format all")
    args = parser.parse_args()

    summary = load_aggregate(args.input)

    if args.format == "latex":
        print(generate_latex_table(summary))
        if args.output:
            Path(args.output).write_text(generate_latex_table(summary))
    elif args.format == "markdown":
        print(generate_markdown_table(summary))
        if args.output:
            Path(args.output).write_text(generate_markdown_table(summary))
    elif args.format == "csv":
        out = args.output or "results.csv"
        export_csv(summary, out)
        print(f"CSV written to {out}")
    elif args.format == "all":
        _export_all(summary, args.output_dir)


if __name__ == "__main__":
    main()
