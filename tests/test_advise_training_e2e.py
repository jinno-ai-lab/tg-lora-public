"""E2E integration tests for advise_training.py CLI (TASK-0122).

Generates synthetic run_metrics.jsonl files, executes the CLI via subprocess,
and validates exit codes, JSON output fields, and advisory correctness.

Exit code semantics:
  0 — normal (healthy or warning)
  1 — file not found / no records
  2 — critical health state
"""

import json
import subprocess
import sys
from pathlib import Path


CLI = Path("scripts/advise_training.py")
ROOT = Path(__file__).resolve().parent.parent

PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(ROOT),
    )


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _converging_records(n: int = 15) -> list[dict]:
    """Steadily decreasing loss with mild noise → converging / healthy."""
    return [
        {
            "type": "cycle_step",
            "cycle": i,
            "loss_train": round(2.5 - 0.12 * i + 0.005 * (i % 3), 4),
            "loss_valid": round(2.4 - 0.11 * i + 0.005 * (i % 3), 4),
            "grad_norm": round(max(0.1, 1.0 - 0.06 * i), 4),
            "velocity_magnitude": round(0.1 * (n - i) / n, 4),
            "tg_lora_accepted": i % 3 != 0,
        }
        for i in range(n)
    ]


def _spike_records(n: int = 10) -> list[dict]:
    """Gradual decrease then sudden loss spike → divergence / warning."""
    records = [
        {"type": "cycle_step", "cycle": i, "loss_train": round(2.0 - 0.08 * i, 4)}
        for i in range(n - 1)
    ]
    # Previous loss ~1.28; spike to 5.0 gives ratio ~3.9x (>= threshold 2.0).
    records.append({"type": "cycle_step", "cycle": n - 1, "loss_train": 5.0})
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdviseTrainingE2E:
    """End-to-end tests exercising advise_training.py as a subprocess."""

    # -- exit code tests ---------------------------------------------------

    def test_e2e_normal_run(self, tmp_path: Path):
        """Converging data → exit 0, healthy health."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _converging_records())
        r = _run_cli(str(jsonl), "--json")
        assert r.returncode == 0, f"stderr: {r.stderr}"
        data = json.loads(r.stdout)
        assert data["overall_health"] == "healthy"

    def test_e2e_missing_file(self, tmp_path: Path):
        """Non-existent path → exit 1."""
        r = _run_cli(str(tmp_path / "nope.jsonl"), "--json")
        assert r.returncode == 1

    def test_e2e_empty_file(self, tmp_path: Path):
        """Empty JSONL → exit 1 (no records)."""
        (tmp_path / "empty.jsonl").write_text("")
        r = _run_cli(str(tmp_path / "empty.jsonl"), "--json")
        assert r.returncode == 1

    def test_e2e_spike_warning_run(self, tmp_path: Path):
        """Loss spike → exit 0 (training can continue), overall_health = warning."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _spike_records())
        r = _run_cli(str(jsonl), "--json")
        assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"
        data = json.loads(r.stdout)
        assert data["overall_health"] == "warning"

    # -- JSON output structure ---------------------------------------------

    def test_e2e_json_output_fields(self, tmp_path: Path):
        """JSON output contains all required top-level and action fields."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _converging_records())
        r = _run_cli(str(jsonl), "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)

        for key in ("overall_health", "actions", "summary",
                     "cycle_health", "trajectory_summary"):
            assert key in data, f"Missing top-level field: {key}"

        assert isinstance(data["actions"], list)
        assert len(data["actions"]) >= 1
        for a in data["actions"]:
            for k in ("action_type", "priority", "reason", "confidence",
                       "suggested_value"):
                assert k in a, f"Action missing field: {k}"
            assert a["priority"] in PRIORITY_RANK

    def test_e2e_actions_priority_order(self, tmp_path: Path):
        """Actions are emitted in non-decreasing priority order."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _spike_records())
        r = _run_cli(str(jsonl), "--json")
        data = json.loads(r.stdout)
        ranks = [PRIORITY_RANK[a["priority"]] for a in data["actions"]]
        for i in range(1, len(ranks)):
            assert ranks[i] >= ranks[i - 1], (
                f"Actions out of priority order: "
                f"{[a['priority'] for a in data['actions']]}"
            )

    # -- output file (-o) --------------------------------------------------

    def test_e2e_output_file(self, tmp_path: Path):
        """-o flag writes JSON report to the specified path."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _converging_records())
        out = tmp_path / "out" / "report.json"
        r = _run_cli(str(jsonl), "--json", "-o", str(out))
        assert r.returncode == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "overall_health" in data

    # -- trajectory quality -------------------------------------------------

    def test_e2e_converging_trajectory(self, tmp_path: Path):
        """Converging data → healthy + positive convergence_rate in trajectory."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _converging_records())
        r = _run_cli(str(jsonl), "--json")
        data = json.loads(r.stdout)
        assert data["overall_health"] == "healthy"
        ts = data["trajectory_summary"]
        assert ts is not None, "trajectory_summary should be present (15 points)"
        assert ts["convergence_rate"] > 0, "Converging data should have positive rate"

    # -- text output --------------------------------------------------------

    def test_e2e_text_output_format(self, tmp_path: Path):
        """Default text output contains expected sections."""
        jsonl = _write_jsonl(tmp_path / "m.jsonl", _converging_records())
        r = _run_cli(str(jsonl))
        assert r.returncode == 0
        assert "Training Advisory Report" in r.stdout
        assert "Overall Health" in r.stdout
        assert "Recommended Actions" in r.stdout
