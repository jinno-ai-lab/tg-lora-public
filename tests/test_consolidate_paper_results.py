"""Tests for scripts/consolidate_paper_results.py — Paper results consolidation."""
from __future__ import annotations

import json
import sys
from pathlib import Path


from scripts.consolidate_paper_results import (
    _find_sibling_json,
    _gate_outcome,
    build_consolidated_report,
    claim_blockers,
    determine_claim_level,
    generate_latex_table,
    generate_markdown_table,
)


def _make_gate_result(gate: str, name: str, passed: bool) -> dict:
    return {"gate": gate, "name": name, "passed": passed, "checks": []}


def _make_gates(**overrides: bool) -> list[dict]:
    defaults = {"G0": True, "G1": True, "G2": False, "G3": False, "G4": False}
    defaults.update(overrides)
    names = {
        "G0": "Hygiene",
        "G1": "Replicated Internal Efficiency",
        "G2": "Memory Frontier Separation",
        "G3": "External Quality Retention",
        "G4": "Causal Attribution",
    }
    return [_make_gate_result(g, names[g], defaults[g]) for g in ["G0", "G1", "G2", "G3", "G4"]]


def _make_summary(
    *,
    n_seeds: int = 3,
    tg_eff: float = 4.0,
    bl_eff: float = 2.0,
    tg_loss: float = 2.5,
    bl_loss: float = 2.5,
    tg_peak: float = 6000.0,
    bl_peak: float = 8000.0,
    freed_mb: float = 2000.0,
) -> dict:

    seeds = list(range(42, 42 + n_seeds))
    per_seed = []
    for s in seeds:
        per_seed.append({
            "seed": s,
            "warm_tg_loss_red_per_wall_minute": tg_eff,
            "warm_baseline_loss_red_per_wall_minute": bl_eff,
            "warm_tg_best_valid_loss": tg_loss,
            "warm_baseline_best_valid_loss": bl_loss,
            "warm_tg_gpu_peak_mb": tg_peak,
            "warm_baseline_gpu_peak_mb": bl_peak,
            "warm_tg_runtime_offload_gpu_freed_mb": freed_mb,
        })

    def _agg(val):
        return {"values": [val] * n_seeds, "mean": val, "stdev": 0.0}

    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {
            "warm_tg_loss_red_per_wall_minute": _agg(tg_eff),
            "warm_baseline_loss_red_per_wall_minute": _agg(bl_eff),
            "warm_tg_best_valid_loss": _agg(tg_loss),
            "warm_baseline_best_valid_loss": _agg(bl_loss),
            "warm_tg_gpu_peak_mb": _agg(tg_peak),
            "warm_baseline_gpu_peak_mb": _agg(bl_peak),
            "warm_tg_runtime_offload_gpu_freed_mb": _agg(freed_mb),
        },
    }


def _make_frontier(
    *,
    boundary: int | None = 3072,
    detected: bool = True,
    avg_savings: float | None = 25.0,
) -> dict:
    return {
        "generated_at": "2026-05-25T00:00:00+00:00",
        "seq_lens": [1536, 2048, 3072],
        "runs": [
            {"seq_len": 1536, "baseline_status": "completed", "tg_status": "completed", "frontier_separation": False},
            {"seq_len": 2048, "baseline_status": "completed", "tg_status": "completed", "frontier_separation": False},
            {
                "seq_len": 3072,
                "baseline_status": "oom",
                "tg_status": "completed",
                "frontier_separation": detected,
                "memory_delta_mb": 2000.0,
                "baseline_peak_mb": 8000.0,
                "tg_peak_mb": 6000.0,
            },
        ],
        "frontier_boundary": boundary,
        "frontier_separation_detected": detected,
        "avg_memory_savings_pct": avg_savings,
    }


class TestDetermineClaimLevel:
    def test_none_when_no_gates_pass(self):
        gates = _make_gates(G0=False, G1=False)
        assert determine_claim_level(gates) == "none"

    def test_c0_when_g1_only(self):
        gates = _make_gates(G1=True, G2=False, G3=False)
        assert determine_claim_level(gates) == "C0"

    def test_c1_when_g1_and_g3(self):
        gates = _make_gates(G1=True, G2=False, G3=True)
        assert determine_claim_level(gates) == "C1"

    def test_c2_when_g1_g2_g3(self):
        gates = _make_gates(G1=True, G2=True, G3=True)
        assert determine_claim_level(gates) == "C2"

    def test_c0_not_c1_when_g1_and_g2_but_not_g3(self):
        gates = _make_gates(G1=True, G2=True, G3=False)
        assert determine_claim_level(gates) == "C0"

    def test_empty_gates(self):
        assert determine_claim_level([]) == "none"


class TestGenerateMarkdownTable:
    def test_basic_structure(self):
        gates = _make_gates(G1=True)
        md = generate_markdown_table(gates, None, None)
        assert "# Paper Results Summary" in md
        assert "Claim Level" in md
        assert "C0" in md

    def test_gate_overview_table(self):
        gates = _make_gates(G1=True, G2=False)
        md = generate_markdown_table(gates, None, None)
        assert "PASS" in md
        assert "FAIL" in md

    def test_includes_aggregate_metrics(self):
        gates = _make_gates(G1=True)
        summary = _make_summary()
        md = generate_markdown_table(gates, summary, None)
        assert "Aggregate Metrics" in md
        assert "4.0000" in md  # tg_eff value

    def test_includes_frontier_separation(self):
        gates = _make_gates(G1=True, G2=True)
        frontier = _make_frontier()
        md = generate_markdown_table(gates, None, frontier)
        assert "Frontier Separation" in md
        assert "3072" in md
        assert "25.0%" in md

    def test_frontier_per_seq_table(self):
        gates = _make_gates()
        frontier = _make_frontier()
        md = generate_markdown_table(gates, None, frontier)
        assert "Per-Sequence Results" in md
        assert "1536" in md
        assert "2000 MB" in md

    def test_claim_ladder_explanation(self):
        gates = _make_gates(G1=True)
        md = generate_markdown_table(gates, None, None)
        assert "C0" in md
        assert "C1" in md
        assert "C2" in md


class TestGenerateLatexTable:
    def test_basic_latex_structure(self):
        gates = _make_gates(G1=True)
        summary = _make_summary()
        latex = generate_latex_table(gates, summary)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "\\begin{tabular}" in latex

    def test_metrics_populated(self):
        gates = _make_gates(G1=True)
        summary = _make_summary()
        latex = generate_latex_table(gates, summary)
        assert "4.00" in latex  # tg_eff
        assert "2.00" in latex  # bl_eff

    def test_no_summary_returns_comment(self):
        gates = _make_gates()
        latex = generate_latex_table(gates, None)
        assert latex.startswith("%")

    def test_claim_level_in_table(self):
        gates = _make_gates(G1=True, G3=True)
        summary = _make_summary()
        latex = generate_latex_table(gates, summary)
        assert "Strong" in latex  # C1 label

    def test_c2_label(self):
        gates = _make_gates(G1=True, G2=True, G3=True)
        summary = _make_summary()
        latex = generate_latex_table(gates, summary)
        assert "Revolutionary" in latex


class TestBuildConsolidatedReport:
    def test_basic_structure(self):
        gates = _make_gates(G1=True)
        report = build_consolidated_report(gates, None, None)
        assert "generated_at" in report
        assert report["claim_level"] == "C0"
        assert "gates" in report
        assert "gate_summary" in report

    def test_gate_summary_counts(self):
        gates = _make_gates(G0=True, G1=True, G2=False, G3=False, G4=False)
        report = build_consolidated_report(gates, None, None)
        assert report["gate_summary"]["total"] == 5
        assert report["gate_summary"]["passed"] == 2
        assert report["gate_summary"]["failed"] == 3

    def test_includes_aggregate_metrics(self):
        gates = _make_gates(G1=True)
        summary = _make_summary()
        report = build_consolidated_report(gates, summary, None)
        assert "aggregate_metrics" in report
        assert report["n_seeds"] == 3

    def test_includes_frontier_data(self):
        gates = _make_gates(G1=True, G2=True)
        frontier = _make_frontier()
        report = build_consolidated_report(gates, None, frontier)
        assert "frontier" in report
        assert report["frontier"]["boundary"] == 3072
        assert report["frontier"]["n_seq_lens"] == 3

    def test_claim_level_c2(self):
        gates = _make_gates(G1=True, G2=True, G3=True)
        report = build_consolidated_report(gates, None, None)
        assert report["claim_level"] == "C2"

    def test_claim_descriptions_present(self):
        gates = _make_gates(G1=True)
        report = build_consolidated_report(gates, None, None)
        assert "C0" in report["claim_descriptions"]
        assert "C1" in report["claim_descriptions"]
        assert "C2" in report["claim_descriptions"]


class TestInsufficientEvidenceHonesty:
    """Consolidate must distinguish an unmeasured gate (INSUFFICIENT) from a
    disproven one (FAIL) — the TASK-0144 third state propagated into the
    claim-level report. A gate whose evidence was never produced must not read
    as a refuted claim."""

    @staticmethod
    def _gate(gate: str, name: str, *, passed: bool, evaluated: bool = True) -> dict:
        result: dict = {"gate": gate, "name": name, "passed": passed, "checks": []}
        if not evaluated:
            result["evaluated"] = False
        return result

    def test_gate_outcome_three_states(self):
        assert _gate_outcome({"passed": True}) == "pass"
        assert _gate_outcome({"passed": False}) == "fail"
        assert _gate_outcome({"passed": False, "evaluated": False}) == "insufficient"

    def test_evaluated_absent_is_fail_not_insufficient(self):
        # Legacy gate dicts (no evaluated key) stay FAIL — not silently upgraded
        # to INSUFFICIENT. Byte-identity with pre-TASK-0144 reports.
        assert _gate_outcome({"passed": False}) == "fail"
        assert _gate_outcome({"passed": True}) == "pass"

    def test_consolidated_report_carries_status_and_counts(self):
        gates = [
            self._gate("G0", "Hygiene", passed=True),
            self._gate("G1", "Efficiency", passed=True),
            self._gate("G2", "Frontier", passed=False),  # disproven
            self._gate("G3", "Quality", passed=False, evaluated=False),  # unmeasured
            self._gate("G4", "Causal", passed=False, evaluated=False),  # unmeasured
        ]
        report = build_consolidated_report(gates, None, None)
        gs = report["gate_summary"]
        assert gs["insufficient"] == 2  # G3, G4
        assert gs["disproven"] == 1  # G2 only — disproven excludes unmeasured
        assert gs["failed"] == 3  # backward-compat: disproven + insufficient
        assert gs["passed"] == 2
        # per-gate entries expose the flag + outcome
        assert report["gates"]["G3"]["status"] == "insufficient"
        assert report["gates"]["G3"]["evaluated"] is False
        assert report["gates"]["G2"]["status"] == "fail"
        assert report["gates"]["G2"]["evaluated"] is True

    def test_byte_identity_when_all_evaluated(self):
        # The historical case: no evaluated=False gates -> disproven == failed
        # and insufficient == 0. Existing reports unchanged.
        gates = _make_gates(G0=True, G1=True, G2=False, G3=False, G4=False)
        gs = build_consolidated_report(gates, None, None)["gate_summary"]
        assert gs["disproven"] == gs["failed"] == 3
        assert gs["insufficient"] == 0

    def test_claim_blockers_insufficient_not_disproven(self):
        # G1 passed -> C0 reached; next claim C1 needs G3. G3 unmeasured -> it
        # must classify as INSUFFICIENT, never disproven. Core honesty: "not yet
        # measured" != "refuted".
        gates = [
            self._gate("G0", "Hygiene", passed=True),
            self._gate("G1", "Efficiency", passed=True),
            self._gate("G3", "Quality", passed=False, evaluated=False),
        ]
        blockers = claim_blockers(gates)
        assert blockers["achieved"] == "C0"
        assert blockers["next"] == "C1"
        assert blockers["insufficient"] == ["G3"]
        assert blockers["disproven"] == []

    def test_claim_blockers_g2_insufficient_blocks_c2_not_disproven(self):
        # G1 + G3 pass -> C1 reached; the top claim C2 (frontier separation) needs
        # G2. An un-run frontier sweep makes G2 INSUFFICIENT, so C2 must be blocked
        # by an UNMEASURED G2, never a disproven one — a not-yet-run revolutionary
        # claim cannot read as refuted. This is the C2-analog of the G3/C1 case
        # above and the consolidate-side proof of the _check_g2 fix: once
        # _check_g2 emits evaluated=False, a real no-frontier gate_report.json
        # flows here verbatim and the top claim reports honestly.
        gates = [
            self._gate("G0", "Hygiene", passed=True),
            self._gate("G1", "Efficiency", passed=True),
            self._gate("G2", "Frontier", passed=False, evaluated=False),  # un-run sweep
            self._gate("G3", "Quality", passed=True),
        ]
        blockers = claim_blockers(gates)
        assert blockers["achieved"] == "C1"
        assert blockers["next"] == "C2"
        assert blockers["insufficient"] == ["G2"]
        assert blockers["disproven"] == []

    def test_claim_blockers_disproven_not_insufficient(self):
        gates = [
            self._gate("G0", "Hygiene", passed=True),
            self._gate("G1", "Efficiency", passed=True),
            self._gate("G3", "Quality", passed=False),  # evaluated, refuted
        ]
        blockers = claim_blockers(gates)
        assert blockers["disproven"] == ["G3"]
        assert blockers["insufficient"] == []

    def test_claim_blockers_top_claim_has_no_next(self):
        gates = [
            self._gate("G1", "Efficiency", passed=True),
            self._gate("G2", "Frontier", passed=True),
            self._gate("G3", "Quality", passed=True),
        ]
        blockers = claim_blockers(gates)
        assert blockers["achieved"] == "C2"
        assert blockers["next"] is None
        assert blockers["disproven"] == []
        assert blockers["insufficient"] == []

    def test_reversal_is_data_driven(self):
        # Toggling evaluated flips disproven <-> insufficient for the same gate.
        disproven = [
            self._gate("G0", "Hygiene", passed=True),
            self._gate("G1", "Efficiency", passed=True),
            self._gate("G3", "Quality", passed=False),
        ]
        insufficient = [
            self._gate("G0", "Hygiene", passed=True),
            self._gate("G1", "Efficiency", passed=True),
            self._gate("G3", "Quality", passed=False, evaluated=False),
        ]
        assert claim_blockers(disproven)["disproven"] == ["G3"]
        assert claim_blockers(insufficient)["insufficient"] == ["G3"]

    def test_markdown_three_state_overview(self):
        gates = [
            self._gate("G0", "Hygiene", passed=True),
            self._gate("G1", "Efficiency", passed=True),
            self._gate("G2", "Frontier", passed=False),  # FAIL
            self._gate("G3", "Quality", passed=False, evaluated=False),  # INSUFFICIENT
        ]
        md = generate_markdown_table(gates, None, None)
        assert "INSUFFICIENT" in md
        assert "FAIL" in md
        assert "PASS" in md

    def test_markdown_disproven_blocker_line(self):
        # G1 passes (C0); next claim C1 needs G3, which was tested and refuted.
        gates = [
            self._gate("G0", "Hygiene", passed=True),
            self._gate("G1", "Efficiency", passed=True),
            self._gate("G3", "Quality", passed=False),  # disproven
        ]
        md = generate_markdown_table(gates, None, None)
        assert "blocked by disproven evidence: G3" in md
        assert "pending evidence" not in md

    def test_markdown_pending_evidence_line(self):
        # Same structure, but G3 was never measured -> pending, not disproven.
        gates = [
            self._gate("G0", "Hygiene", passed=True),
            self._gate("G1", "Efficiency", passed=True),
            self._gate("G3", "Quality", passed=False, evaluated=False),
        ]
        md = generate_markdown_table(gates, None, None)
        assert "pending evidence" in md
        assert "blocked by disproven evidence" not in md

    def test_claim_blockers_absent_gate_is_insufficient(self):
        # A required gate entirely missing from the results is evidence never
        # produced -> insufficient, not disproven.
        gates = [
            self._gate("G0", "Hygiene", passed=True),
            self._gate("G1", "Efficiency", passed=True),
            # G3 (required for C1) is absent
        ]
        blockers = claim_blockers(gates)
        assert blockers["next"] == "C1"
        assert blockers["insufficient"] == ["G3"]
        assert blockers["disproven"] == []

    def test_markdown_byte_identity_when_all_evaluated(self):
        gates = _make_gates(G0=True, G1=True, G2=False, G3=False, G4=False)
        md = generate_markdown_table(gates, None, None)
        assert "INSUFFICIENT" not in md


class TestCLIIntegration:
    def test_full_pipeline(self, tmp_path):
        gates = _make_gates(G0=True, G1=True, G2=False, G3=False, G4=False)
        gate_data = {
            "gates": gates,
            "overall_passed": False,
            "generated_at": "2026-05-25T00:00:00+00:00",
        }
        gate_path = tmp_path / "gate_report.json"
        gate_path.write_text(json.dumps(gate_data))

        summary = _make_summary()
        summary_path = tmp_path / "aggregate_summary.json"
        summary_path.write_text(json.dumps(summary))

        frontier = _make_frontier(detected=False, boundary=None)
        frontier_path = tmp_path / "frontier_report.json"
        frontier_path.write_text(json.dumps(frontier))

        out_dir = tmp_path / "output"

        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/consolidate_paper_results.py",
                "--gate-report", str(gate_path),
                "--summary", str(summary_path),
                "--frontier-report", str(frontier_path),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, result.stderr
        assert "Claim Level: C0" in result.stdout

        # Check output files exist
        assert (out_dir / "paper_results_summary.md").exists()
        assert (out_dir / "paper_results_table.tex").exists()
        assert (out_dir / "consolidated_report.json").exists()

        # Verify JSON content
        report = json.loads((out_dir / "consolidated_report.json").read_text())
        assert report["claim_level"] == "C0"
        assert report["n_seeds"] == 3

    def test_gate_report_only(self, tmp_path):
        gates = _make_gates(G1=True, G3=True)
        gate_data = {"gates": gates, "overall_passed": True, "generated_at": "2026-05-25T00:00:00+00:00"}
        gate_path = tmp_path / "gate_report.json"
        gate_path.write_text(json.dumps(gate_data))

        out_dir = tmp_path / "output"

        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/consolidate_paper_results.py",
                "--gate-report", str(gate_path),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "Claim Level: C1" in result.stdout

    def test_insufficient_gate_flows_through_cli(self, tmp_path):
        # evaluate_paper_gates stamps evaluated=False when an input is missing;
        # consolidate must carry that through as INSUFFICIENT, not FAIL.
        gates = [
            {"gate": "G0", "name": "Hygiene", "passed": True, "checks": []},
            {"gate": "G1", "name": "Replicated Internal Efficiency", "passed": True, "checks": []},
            {"gate": "G2", "name": "Memory Frontier Separation", "passed": True, "checks": []},
            {"gate": "G3", "name": "External Quality Retention",
             "passed": False, "evaluated": False, "checks": []},
        ]
        gate_data = {
            "gates": gates,
            "overall_passed": False,
            "generated_at": "2026-05-25T00:00:00+00:00",
        }
        gate_path = tmp_path / "gate_report.json"
        gate_path.write_text(json.dumps(gate_data))
        out_dir = tmp_path / "output"

        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/consolidate_paper_results.py",
                "--gate-report", str(gate_path),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, result.stderr
        assert "INSUFFICIENT" in result.stdout

        report = json.loads((out_dir / "consolidated_report.json").read_text())
        assert report["gate_summary"]["insufficient"] == 1
        assert report["gate_summary"]["disproven"] == 0
        assert report["claim_blockers"]["insufficient"] == ["G3"]

    def test_missing_gate_report_exits(self, tmp_path):
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/consolidate_paper_results.py",
                "--gate-report", str(tmp_path / "nonexistent.json"),
                "--output-dir", str(tmp_path / "output"),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 2


class TestFindSiblingJson:
    def test_finds_file_same_directory(self, tmp_path):
        (tmp_path / "gate_report.json").write_text("{}")
        (tmp_path / "aggregate_summary.json").write_text('{"seeds": []}')
        result = _find_sibling_json(str(tmp_path / "gate_report.json"), "aggregate_summary.json")
        assert result is not None
        assert result.name == "aggregate_summary.json"

    def test_finds_file_parent_directory(self, tmp_path):
        subdir = tmp_path / "suite"
        subdir.mkdir()
        (subdir / "gate_report.json").write_text("{}")
        (tmp_path / "aggregate_summary.json").write_text('{"seeds": []}')
        result = _find_sibling_json(str(subdir / "gate_report.json"), "aggregate_summary.json")
        assert result is not None
        assert result.parent == tmp_path

    def test_finds_frontier_report(self, tmp_path):
        (tmp_path / "gate_report.json").write_text("{}")
        (tmp_path / "frontier_report.json").write_text('{"runs": []}')
        result = _find_sibling_json(str(tmp_path / "gate_report.json"), "frontier_report.json")
        assert result is not None
        assert result.name == "frontier_report.json"

    def test_returns_none_when_not_found(self, tmp_path):
        (tmp_path / "gate_report.json").write_text("{}")
        result = _find_sibling_json(str(tmp_path / "gate_report.json"), "aggregate_summary.json")
        assert result is None

    def test_does_not_search_beyond_grandparent(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "gate_report.json").write_text("{}")
        (tmp_path / "aggregate_summary.json").write_text('{"seeds": []}')
        result = _find_sibling_json(str(deep / "gate_report.json"), "aggregate_summary.json")
        assert result is None


class TestAutoDiscoveryIntegration:
    def test_auto_discovers_summary_and_frontier(self, tmp_path):
        """Without explicit --summary/--frontier-report, auto-discovers from gate_report location."""
        gates = _make_gates(G0=True, G1=True, G2=True, G3=False, G4=False)
        gate_data = {"gates": gates, "overall_passed": False, "generated_at": "2026-05-25T00:00:00+00:00"}
        gate_path = tmp_path / "gate_report.json"
        gate_path.write_text(json.dumps(gate_data))

        summary = _make_summary()
        (tmp_path / "aggregate_summary.json").write_text(json.dumps(summary))

        frontier = _make_frontier(detected=True, boundary=3072)
        (tmp_path / "frontier_report.json").write_text(json.dumps(frontier))

        out_dir = tmp_path / "output"

        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/consolidate_paper_results.py",
                "--gate-report", str(gate_path),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, result.stderr
        assert "Auto-discovered aggregate summary" in result.stdout
        assert "Auto-discovered frontier report" in result.stdout

        report = json.loads((out_dir / "consolidated_report.json").read_text())
        assert report["claim_level"] == "C0"
        assert "aggregate_metrics" in report
        assert "frontier" in report

    def test_explicit_paths_override_auto_discovery(self, tmp_path):
        """Explicit --summary/--frontier-report take precedence over auto-discovered files."""
        gates = _make_gates(G0=True, G1=True)
        gate_data = {"gates": gates, "overall_passed": True, "generated_at": "2026-05-25T00:00:00+00:00"}
        gate_path = tmp_path / "gate_report.json"
        gate_path.write_text(json.dumps(gate_data))

        # Place a "stale" summary next to the gate report (auto-discoverable)
        stale = _make_summary(tg_eff=0.001)
        (tmp_path / "aggregate_summary.json").write_text(json.dumps(stale))

        # Place the "real" summary elsewhere
        real = _make_summary(tg_eff=4.0)
        real_path = tmp_path / "real_summary.json"
        real_path.write_text(json.dumps(real))

        out_dir = tmp_path / "output"

        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/consolidate_paper_results.py",
                "--gate-report", str(gate_path),
                "--summary", str(real_path),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, result.stderr
        # Should NOT auto-discover since explicit path was given
        assert "Auto-discovered aggregate summary" not in result.stdout
