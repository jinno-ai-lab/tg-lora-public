from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_cosine_n_ablation import load_run_metrics


def test_load_run_metrics_skips_non_dict_jsonl_lines(tmp_path: Path) -> None:
    # A valid-JSON-but-non-object line (a bare array/scalar from a torn flush or
    # a hand-edited file) parsed fine yet made ``record.get("type")`` raise an
    # uncaught AttributeError that aborted the whole cosine-N ablation summary —
    # the same non-dict-after-loads crash class load_run (374468d) closed. Skip
    # the stray line and degrade; genuine invalid-JSON still raises from
    # orjson.loads (fail-loud preserved, matching load_run's posture).
    p = tmp_path / "metrics.jsonl"
    lines = [
        json.dumps({"type": "run_header", "condition": "cosine_n"}),
        "[1, 2, 3]",  # non-dict line — would crash record.get("type")
        "42",  # another non-dict variant (bare scalar)
        json.dumps({"type": "step", "loss_valid": 1.1, "tg_lora_N": 4}),
        json.dumps({"type": "run_footer", "best_valid_loss": 1.1}),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    header, records, footer = load_run_metrics(p)
    assert header["condition"] == "cosine_n"
    assert footer["best_valid_loss"] == 1.1
    assert len(records) == 1
    assert records[0]["loss_valid"] == 1.1
