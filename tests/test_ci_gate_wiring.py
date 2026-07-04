"""Static guard: the GPU-free velocity-ops CI gate must actually be invoked.

Closes the loop the AI-Hub feedback named — "the four new gate targets exist as
make entries but there is no evidence they are invoked by CI or the autonomous
loop — confirm at least the GPU-free ... gates are wired into the gate sequence,
else they are inert". On this public mirror the autonomous loop is AI-Hub
infrastructure (not in-repo), so the faithful in-repo realization of "the gate
sequence" is the GitHub Actions workflow. Two invariants are pinned here:

1. ``.github/workflows/test.yml`` invokes the portable velocity-ops gate on every
   push/PR. Before this, ``bench-velocity-ops-ci`` was a make target that
   *nothing* ran — inert by definition.

2. The gate it invokes is the PORTABLE one (``--max-cap-overhead-ratio``), not the
   old non-portable ``--baseline ... --threshold`` absolute-time comparison. That
   absolute comparison is defeated across hardware: it reported a 12x "regression"
   under pytest load on a 12GB box against the checked-in baseline, so it could
   never have been a real CI gate. The within-run capped/no-cap ratio is
   hardware-normalized — see ``tests/test_benchmark_velocity_ops.py`` for the
   both-sides boundary tests that prove it enforces.

If a future change drops the workflow job or reverts the make target to the
absolute-time baseline, this guard fails loud — exactly so the gate cannot
silently regress to "exists but inert".
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
MAKEFILE = REPO_ROOT / "Makefile"


def test_workflow_invokes_portable_velocity_gate() -> None:
    """The GH Actions workflow must invoke the portable velocity-ops gate."""
    assert WORKFLOW.exists(), ".github/workflows/test.yml must exist"
    text = WORKFLOW.read_text()
    assert "benchmark_velocity_ops.py" in text, (
        "velocity-ops benchmark is not invoked by the CI workflow — the "
        "GPU-free gate is inert (nothing runs it)."
    )
    assert "--max-cap-overhead-ratio" in text, (
        "CI must invoke the PORTABLE gate (--max-cap-overhead-ratio), not the "
        "non-portable absolute-time --baseline comparison."
    )


def test_makefile_ci_target_is_portable() -> None:
    """The bench-velocity-ops-ci make target must use the portable ratio gate."""
    assert MAKEFILE.exists(), "Makefile must exist"
    text = MAKEFILE.read_text()
    assert "bench-velocity-ops-ci:" in text, "bench-velocity-ops-ci target must exist"
    # Isolate the target body (up to the next blank line / target).
    body = text.split("bench-velocity-ops-ci:", 1)[1].split("\n\n", 1)[0]
    assert "--max-cap-overhead-ratio" in body, (
        "bench-velocity-ops-ci must use --max-cap-overhead-ratio (portable)."
    )
    assert (
        "--threshold 20" not in body
        and "--baseline baselines/velocity_ops.json" not in body
    ), (
        "bench-velocity-ops-ci must NOT use the non-portable "
        "--baseline/--threshold absolute-time gate."
    )
