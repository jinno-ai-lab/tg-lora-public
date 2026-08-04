"""Pin the GPU-availability probe added to ``scripts/agent_check_status.py``.

The steering feedback (AI_HUB_MAKE_RUN_FEEDBACK) asked — across iterations —
the same question the auto-loop kept answering with another report-tooling
hardening pass instead of resolving: "is the GPU available to start/continue
the 9B run, and if not, what concretely unblocks it?". ``report_gpu_availability``
surfaces that decision to the operator inside ``make status`` (the command they
already run), so the blocked state stops being re-derived each iteration.

Crucially this is INFORMATIONAL, not a status-check failure: a host with no GPU
(CI, a CPU-only dev box, the MLX Track-B path) must still pass ``make status``.
``query_gpu_compute_apps`` therefore returns ``None`` whenever ``nvidia-smi`` is
absent / times out / exits non-zero, and the report degrades to a "cannot
assess" line rather than a non-zero exit.

These tests are DETERMINISTIC — they never invoke the real ``nvidia-smi``. The
nvidia-smi CSV parsing is a pure function (``parse_gpu_holders``) fed fixture
text, and ``report_gpu_availability`` is driven by monkeypatching
``query_gpu_compute_apps`` so the holder-vs-free-vs-absent branches all run
without a GPU present. The headline fixture is the real-world blocker this
probe was built to name: a sibling project's (``recursive_funnel_bitnet``)
``llama-server`` holding the GPU, which is the concrete unblock the operator
needs to see.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "agent_check_status.py"

# The exact CSV ``nvidia-smi --query-compute-apps=pid,process_name,used_memory
# --format=csv,noheader`` produced on the host this probe was added on: the GPU
# was at 94% util, held by a SIBLING project's llama-server — the concrete
# reason the 9B lever was not actionable that cycle.
_RECURSIVE_FUNNEL_CSV = (
    "3563211, /home/jinno/recursive_funnel_bitnet/models/llama-server, 2142 MiB\n"
)


def _load_module():
    """Load the script as an isolated module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("agent_check_status", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_parse_gpu_holders_names_recursive_funnel_bitnet_blocker() -> None:
    """The headline fixture: the CSV observed on the host where this probe was
    added parses to exactly one holder whose name carries the sibling-project
    path the operator must act on."""
    mod = _load_module()
    holders = mod.parse_gpu_holders(_RECURSIVE_FUNNEL_CSV)
    assert len(holders) == 1
    assert holders[0]["pid"] == "3563211"
    assert "recursive_funnel_bitnet" in holders[0]["name"]
    assert "llama-server" in holders[0]["name"]
    assert holders[0]["mem"] == "2142 MiB"


def test_parse_gpu_holders_empty_means_free() -> None:
    """No compute apps (blank / whitespace-only CSV) -> empty list -> the GPU
    reads as free. This is the branch that lets the operator see the 9B lever
    IS actionable when nothing holds the GPU."""
    mod = _load_module()
    assert mod.parse_gpu_holders("") == []
    assert mod.parse_gpu_holders("\n  \n\t\n") == []


def test_parse_gpu_holders_multiple() -> None:
    mod = _load_module()
    csv = (
        "111, /usr/bin/train_a, 1000 MiB\n"
        "222, /usr/bin/train_b, 2000 MiB\n"
    )
    holders = mod.parse_gpu_holders(csv)
    assert [h["pid"] for h in holders] == ["111", "222"]


def test_parse_gpu_holders_skips_malformed_rows() -> None:
    """A row with too few columns (a half-flushed nvidia-smi line) is skipped,
    not raised on — the status check must survive a malformed probe output."""
    mod = _load_module()
    csv = (
        "111, only-two-fields\n"            # malformed — skipped
        "222, /usr/bin/ok, 500 MiB\n"        # valid — kept
        "\n"                                  # blank — skipped
    )
    holders = mod.parse_gpu_holders(csv)
    assert len(holders) == 1
    assert holders[0]["pid"] == "222"


def test_src_data_pipeline_present_false_on_public_mirror() -> None:
    """On this public mirror ``src.data`` is deliberately stripped (DATA/Cat-C),
    so the probe reads False — the contract that drives the DATA-blocked (not
    GPU-blocked) framing of the 9B lever below. Pins that the framing is not
    accidentally inverted by a future port."""
    mod = _load_module()
    assert mod.src_data_pipeline_present() is False


def test_report_gpu_availability_mirror_held_gpu_is_data_blocked_not_gpu(monkeypatch) -> None:
    """The honest contract for THIS public mirror: even when a compute app holds
    the GPU, the report NAMES the holder (the operator still wants to know what
    is on the GPU) but does NOT emit the misleading 'stop the holder to make the
    9B run actionable' — because here the lever is DATA-blocked (src.data
    stripped, Cat-C), and freeing the GPU cannot move it. Asserting 'Stop the
    holder' is ABSENT is the mutation pin: restoring the old GPU-bound copy on
    the mirror re-introduces the false 'free the GPU -> lever actionable' signal
    the loop kept wasting cycles on."""
    mod = _load_module()
    monkeypatch.setattr(mod, "query_gpu_compute_apps", lambda: _RECURSIVE_FUNNEL_CSV)
    monkeypatch.setattr(mod, "src_data_pipeline_present", lambda: False)  # mirror
    import sys
    from io import StringIO

    buf = StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    mod.report_gpu_availability()
    out = buf.getvalue()
    # The holder is still named — the GPU state is informational context.
    assert "recursive_funnel_bitnet" in out
    assert "llama-server" in out
    assert "3563211" in out
    # ... but the conclusion is the honest one: DATA-blocked, GPU won't help.
    assert "DATA-blocked" in out
    assert "will NOT move the lever" in out
    assert "Stop the holder" not in out  # the misleading GPU-bound copy is gone


def test_report_gpu_availability_private_checkout_held_gpu_is_gpu_blocked(monkeypatch) -> None:
    """The OTHER context — the PRIVATE checkout where ``src.data`` ships — is the
    one where the 9B lever really IS GPU-bound, so naming the holder + printing
    'stop the holder to make actionable' is the correct, non-misleading signal.
    Injected via monkeypatch so the test does not depend on a private tree."""
    mod = _load_module()
    monkeypatch.setattr(mod, "query_gpu_compute_apps", lambda: _RECURSIVE_FUNNEL_CSV)
    monkeypatch.setattr(mod, "src_data_pipeline_present", lambda: True)  # private checkout
    import sys
    from io import StringIO

    buf = StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    mod.report_gpu_availability()
    out = buf.getvalue()
    assert "recursive_funnel_bitnet" in out
    assert "3563211" in out
    assert "Stop the holder" in out  # GPU-bound here: freeing it DOES move the lever
    assert "DATA-blocked" not in out


def test_report_gpu_availability_free_gpu(monkeypatch) -> None:
    """No holders -> the report says the GPU looks free. On the mirror it
    additionally states the lever stays DATA-blocked (free GPU does not make it
    actionable here); in the private checkout it states the lever IS actionable."""
    mod = _load_module()
    monkeypatch.setattr(mod, "query_gpu_compute_apps", lambda: "")
    import sys
    from io import StringIO

    buf = StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    mod.report_gpu_availability()
    out = buf.getvalue()
    assert "FREE" in out
    assert "DATA-blocked" in out  # mirror: free GPU still does not move the lever


def test_report_gpu_availability_free_gpu_private_checkout_actionable(monkeypatch) -> None:
    """Private checkout + free GPU -> the lever is actionable this cycle (the
    one combination where ``make status`` should say 'go')."""
    mod = _load_module()
    monkeypatch.setattr(mod, "query_gpu_compute_apps", lambda: "")
    monkeypatch.setattr(mod, "src_data_pipeline_present", lambda: True)
    import sys
    from io import StringIO

    buf = StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    mod.report_gpu_availability()
    out = buf.getvalue()
    assert "FREE" in out
    assert "actionable this cycle" in out
    assert "DATA-blocked" not in out


def test_report_gpu_availability_no_nvidia_smi_does_not_fail(monkeypatch) -> None:
    """The reliability contract: a host with no nvidia-smi (CI / CPU-only / MLX
    Track-B) degrades to an informational 'cannot assess' line and returns
    normally — it must NEVER turn ``make status`` red for lacking a GPU."""
    mod = _load_module()
    monkeypatch.setattr(mod, "query_gpu_compute_apps", lambda: None)
    import sys
    from io import StringIO

    buf = StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    mod.report_gpu_availability()  # must not raise
    out = buf.getvalue()
    assert "cannot assess" in out
