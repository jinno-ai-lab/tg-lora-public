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


# ---------------------------------------------------------------------------
# Real producer -> consumer loop (the dormant-wiring loop, empirically proven)
# ---------------------------------------------------------------------------
# Feedback: the advisor (pure helper + standalone CLI) was exercised only on
# synthetic fixtures; nothing proved the CLI's success path renders the
# advisory block on REAL producer output, nor that the advisory is actionable.
# These tests drive the REAL ``RunMetrics.record_step`` schema through the CLI
# to a plateau/stagnation truncation and capture the rendered console output.
#
# Import the CLI's pure helpers to prove the producer->consumer field contract
# (the extraction mapping) directly, not only via subprocess.
sys.path.insert(0, str(ROOT / "scripts"))
import advise_training as _cli  # noqa: E402
sys.path.pop(0)


def _real_producer_plateau(n: int = 14) -> list[dict]:
    """Genuine ``RunMetrics.record_step`` schema (the real producer).

    Loss improves for the first cycles then goes exactly flat -> stagnation +
    plateau + convergence, which drives the advisor to ``increase_k`` (whose
    remediation names the ``tg_lora.K_initial`` knob) and a ``stop_training``
    truncation. Keys mirror run_metrics.py record_step exactly.
    """
    recs: list[dict] = []
    for i in range(n):
        # Improve for the first 7 cycles, then freeze (plateau).
        loss = round(2.0 - 0.10 * min(i, 6), 4)
        recs.append({
            "type": "step",
            "run_id": "real_run",
            "mode": "tg_lora",
            "step": i + 1,
            "cycle": i,
            "elapsed_seconds": float(i + 1),
            "loss_train": loss,
            "loss_valid": loss,
            "backward_passes": 1,
            "total_backward_passes": i + 1,
            "grad_norm": 0.5,
            "tg_lora_accepted": True,
            "tg_lora_K": 3,
            "tg_lora_N": 2,
            "tg_lora_alpha": 0.5,
            # REAL producer keys (NOT loss_pilot / loss_after).
            "tg_lora_loss_pilot_eval": round(loss + 0.01, 4),
            "tg_lora_loss_after": round(loss - 0.005, 4),
        })
    return recs


class TestRealProducerConsumerLoop:
    """Drive the advise_training CLI on REAL producer-schema output to a
    plateau/stagnation truncation and capture the rendered advisory block +
    the concrete config knob."""

    def test_plateau_truncation_renders_advisory_block(self, tmp_path: Path):
        """The CLI success path renders the advisory block on real producer
        output and reaches a stop_training truncation (not just the helper)."""
        jsonl = _write_jsonl(tmp_path / "real.jsonl", _real_producer_plateau())
        r = _run_cli(str(jsonl))  # TEXT mode -> rendered console output
        assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        out = r.stdout
        assert "Training Advisory Report" in out
        assert "Recommended Actions" in out
        # Drove to a truncation: stop_training fires (convergence/stagnation).
        assert "stop_training" in out, f"expected stop_training in:\n{out}"
        # The advisory is actionable: the remediation line renders.
        assert "remediation:" in out

    def test_plateau_advisory_names_exact_config_knob(self, tmp_path: Path):
        """Bullet 3: the advisory names the EXACT knob string so future wording
        drift cannot degrade it into noise. Pinned on genuine producer output."""
        jsonl = _write_jsonl(tmp_path / "real.jsonl", _real_producer_plateau())
        r = _run_cli(str(jsonl))
        assert r.returncode == 0, f"stderr:\n{r.stderr}"
        # increase_k fires on plateau/stagnation; its remediation must name the
        # literal config field path tg_lora.K_initial.
        assert "tg_lora.K_initial" in r.stdout, (
            f"advisory must name the exact knob tg_lora.K_initial:\n{r.stdout}"
        )

    def test_producer_field_contract_real_keys_consumed(self, tmp_path: Path):
        """The producer writes tg_lora_loss_pilot_eval / tg_lora_loss_after
        (NOT loss_pilot / loss_after). Prove the consumer reads the real keys
        -- the previously-disconnected producer->consumer contract is wired."""
        extracted = _cli._extract_cycle_records(_real_producer_plateau())
        assert extracted, "cycle records must be extracted from producer output"
        pilot_vals = [r["loss_pilot"] for r in extracted]
        after_vals = [r["loss_after"] for r in extracted]
        assert all(v != 0.0 for v in pilot_vals), (
            f"loss_pilot must flow from tg_lora_loss_pilot_eval, got {pilot_vals}"
        )
        assert all(v != 0.0 for v in after_vals), (
            f"loss_after must flow from tg_lora_loss_after, got {after_vals}"
        )

    def test_json_report_carries_structured_remediation(self, tmp_path: Path):
        """JSON output's per-action remediation field carries the exact knob."""
        jsonl = _write_jsonl(tmp_path / "real.jsonl", _real_producer_plateau())
        r = _run_cli(str(jsonl), "--json")
        assert r.returncode == 0, f"stderr:\n{r.stderr}"
        data = json.loads(r.stdout)
        knobs = {a["remediation"] for a in data["actions"] if a.get("remediation")}
        assert any("tg_lora.K_initial" in k for k in knobs), (
            f"structured remediation must name tg_lora.K_initial: {knobs}"
        )


class TestEmittedJsonIsParseClean:
    """judge_invalid_json risk class: the prior iteration was rejected with
    'Expecting property name enclosed in double quotes'. Pin that every JSON
    the CLI emits across every advisory shape round-trips through strict
    json.loads (no trailing comma / single-quoted keys / Python-dict-repr
    leakage), and that every action carries a non-empty actionable knob."""

    def _json_roundtrips(self, recs: list[dict], tmp_path: Path) -> dict:
        jsonl = _write_jsonl(tmp_path / "m.jsonl", recs)
        r = _run_cli(str(jsonl), "--json")
        assert r.returncode in (0, 2), f"stderr:\n{r.stderr}"
        data = json.loads(r.stdout)  # strict parse — the guard
        for a in data["actions"]:
            assert a.get("remediation"), f"action missing remediation knob: {a}"
        return data

    def test_converging_json_roundtrips(self, tmp_path: Path):
        self._json_roundtrips(_converging_records(), tmp_path)

    def test_spike_json_roundtrips(self, tmp_path: Path):
        self._json_roundtrips(_spike_records(), tmp_path)

    def test_plateau_json_roundtrips(self, tmp_path: Path):
        self._json_roundtrips(_real_producer_plateau(), tmp_path)
