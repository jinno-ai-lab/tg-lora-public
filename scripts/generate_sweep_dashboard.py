#!/usr/bin/env python
"""Generate a self-contained HTML dashboard for accel param sweep results.

Usage:
    python scripts/generate_sweep_dashboard.py <sweep_dir>
    python scripts/generate_sweep_dashboard.py --ranking-json <path> --output <path>

Reads ranking.json produced by analyze_accel_sweep.py and generates an HTML
dashboard with summary tables, pairwise comparison, and next-action recommendations.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def load_ranking(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("ranking.json must contain a JSON object")
    return data


def _delta_class(delta: float | None) -> str:
    if delta is None:
        return "neutral"
    return "improvement" if delta < 0 else "worse"


def _fmt(val: float | None, fmt: str = ".6f") -> str:
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _pct_bar(value: float, max_abs: float) -> str:
    if max_abs == 0:
        return ""
    pct = min(abs(value) / max_abs * 100, 100)
    color = "#4caf50" if value < 0 else "#f44336"
    return f'<div class="bar" style="width:{pct}%;background:{color}"></div>'


def generate_html(analysis: dict[str, Any]) -> str:
    baseline = analysis.get("baseline", {})
    best = analysis.get("best_run", {})
    pairwise = analysis.get("pairwise", [])
    total_runs = analysis.get("total_runs", 0)

    max_delta = max((abs(p.get("delta_vs_baseline", 0)) for p in pairwise), default=0)

    rows_html = ""
    for p in pairwise:
        delta = p.get("delta_vs_baseline")
        cls = _delta_class(delta)
        rows_html += f"""<tr>
  <td><code>{p.get('run_id', '?')}</code></td>
  <td>{p.get('decay', 'N/A')}</td>
  <td>{p.get('boost', 'N/A')}</td>
  <td>{_fmt(p.get('best_valid_loss'))}</td>
  <td class="{cls}">{delta:+.6f}</td>
  <td>{_fmt(p.get('delta_pct'), '.2f')}%</td>
  <td>{_fmt(p.get('efficiency_per_bp'), '.2e')}</td>
  <td>{_fmt(p.get('loss_red_per_wall_min'), '.5f')}</td>
  <td><div class="bar-container">{_pct_bar(delta or 0, max_delta)}</div></td>
</tr>
"""

    best_id = best.get("run_id", "N/A")
    best_loss = best.get("best_valid_loss")

    next_action = ""
    if pairwise:
        top = pairwise[0]
        delta = top.get("delta_vs_baseline")
        if delta is not None and delta < -0.001:
            next_action = f"""<div class="action improvement">
  <strong>Improvement found:</strong> <code>{top['run_id']}</code> reduces loss by {abs(delta):.4f}.
  Proceed with lm-evaluation-harness on this config.
</div>"""
        elif delta is not None and delta > 0.001:
            next_action = """<div class="action worse">
  <strong>No improvement:</strong> all treatments worse than baseline.
  Consider different accel param ranges or disabling accel adaptation.
</div>"""
        else:
            next_action = """<div class="action neutral">
  <strong>Neutral:</strong> no significant difference from baseline.
  Consider expanding parameter grid or increasing training cycles.
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Accel Param Sweep Dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 2rem auto; max-width: 960px; color: #333; background: #fafafa; }}
h1 {{ border-bottom: 2px solid #1976d2; padding-bottom: .5rem; }}
h2 {{ color: #1976d2; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ padding: .5rem .75rem; text-align: right; border-bottom: 1px solid #e0e0e0; }}
th {{ background: #f5f5f5; font-weight: 600; text-align: left; }}
td:first-child {{ text-align: left; }}
code {{ background: #f0f0f0; padding: .15rem .3rem; border-radius: 3px; }}
.improvement {{ color: #2e7d32; font-weight: 600; }}
.worse {{ color: #c62828; font-weight: 600; }}
.neutral {{ color: #666; }}
.bar-container {{ position: relative; height: 1.2rem; background: #eee; border-radius: 3px; min-width: 80px; }}
.bar {{ height: 100%; border-radius: 3px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: 1rem; margin: 1rem 0; }}
.card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem; }}
.card h3 {{ margin: 0 0 .5rem; font-size: .9rem; color: #666; text-transform: uppercase; }}
.card .value {{ font-size: 1.4rem; font-weight: 700; }}
.action {{ padding: 1rem; border-radius: 6px; margin: 1rem 0; }}
.action.improvement {{ background: #e8f5e9; border: 1px solid #4caf50; }}
.action.worse {{ background: #fbe9e7; border: 1px solid #f44336; }}
.action.neutral {{ background: #f5f5f5; border: 1px solid #bdbdbd; }}
footer {{ margin-top: 2rem; color: #999; font-size: .85rem; border-top: 1px solid #eee; padding-top: .5rem; }}
</style>
</head>
<body>
<h1>Accel Param Sweep Dashboard</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &mdash; {total_runs} configs analyzed</p>

<div class="summary-grid">
  <div class="card">
    <h3>Best Config</h3>
    <div class="value"><code>{best_id}</code></div>
  </div>
  <div class="card">
    <h3>Best Valid Loss</h3>
    <div class="value">{_fmt(best_loss)}</div>
  </div>
  <div class="card">
    <h3>Baseline Loss</h3>
    <div class="value">{_fmt(baseline.get('best_valid_loss'))}</div>
  </div>
  <div class="card">
    <h3>Baseline Wall Time</h3>
    <div class="value">{_fmt(baseline.get('wall_seconds'), '.0f')}s</div>
  </div>
</div>

<h2>Pairwise Comparison vs Baseline (no_accel)</h2>
<table>
<thead>
<tr>
  <th>Config</th><th>decay</th><th>boost</th><th>Best Loss</th>
  <th>&Delta; vs baseline</th><th>&Delta;%</th><th>Eff/bp</th>
  <th>Loss red/min</th><th>Visual</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

<h2>Next Actions</h2>
{next_action}

<footer>Generated by <code>generate_sweep_dashboard.py</code> &mdash; TASK-0094</footer>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML dashboard for accel param sweep results",
    )
    parser.add_argument(
        "sweep_dir", nargs="?",
        help="Sweep directory containing analysis/ranking.json",
    )
    parser.add_argument(
        "--ranking-json", default="",
        help="Explicit path to ranking.json (overrides sweep_dir)",
    )
    parser.add_argument(
        "--output", default="",
        help="Output HTML path (defaults to sweep_dir/analysis/dashboard.html)",
    )
    args = parser.parse_args()

    if args.ranking_json:
        ranking_path = Path(args.ranking_json)
    elif args.sweep_dir:
        ranking_path = Path(args.sweep_dir) / "analysis" / "ranking.json"
    else:
        parser.error("Provide sweep_dir or --ranking-json")

    if not ranking_path.exists():
        print(f"Error: {ranking_path} not found", file=sys.stderr)
        sys.exit(1)

    analysis = load_ranking(ranking_path)

    if args.output:
        output_path = Path(args.output)
    elif args.sweep_dir:
        output_path = Path(args.sweep_dir) / "analysis" / "dashboard.html"
    else:
        output_path = ranking_path.parent / "dashboard.html"

    html = generate_html(analysis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    main()
