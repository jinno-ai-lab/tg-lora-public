#!/usr/bin/env python3
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

def count_lines(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def check_datasets():
    print("=== [1/3] Dataset Integrity Check ===")
    data_dir = Path("data")
    expected = {
        "train.jsonl": 4500,
        "valid_quick.jsonl": 450,
        "test.jsonl": 450,
    }
    
    all_ok = True
    for filename, min_lines in expected.items():
        path = data_dir / filename
        if not path.exists():
            print(f"[-] {filename}: MISSING")
            all_ok = False
        else:
            lines = count_lines(path)
            if lines < min_lines:
                print(f"[-] {filename}: Present but short ({lines}/{min_lines} lines)")
                all_ok = False
            else:
                print(f"[+] {filename}: OK ({lines} lines)")
                
    # Check MLX data link
    mlx_dir = Path("data_mlx")
    if mlx_dir.exists():
        train_link = mlx_dir / "train.jsonl"
        valid_link = mlx_dir / "valid.jsonl"
        if not train_link.exists() or not valid_link.exists():
            print("[-] MLX symlinks: INCOMPLETE or MISSING")
            all_ok = False
        else:
            print("[+] MLX symlinks: OK")
    else:
        print("[ ] MLX data directory not found (Track B not configured on this machine)")
        
    return all_ok

def check_experiment_runs():
    print("\n=== [2/3] Experiment Runs & Aggregate Summaries ===")
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("[-] runs/ directory does not exist.")
        return None
        
    # Look for aggregate summaries or paper-memory suites
    suites = list(runs_dir.glob("paper_memory_suite_*"))
    one_shot_suites = list(runs_dir.glob("paper_memory_one_shot_suite_*"))
    all_suites = sorted(suites + one_shot_suites, key=os.path.getmtime, reverse=True)
    
    if not all_suites:
        print("[-] No paper-memory suites found in runs/")
        return None
        
    print(f"Found {len(all_suites)} paper-memory suite runs. Most recent:")
    most_recent = all_suites[0]
    print(f"  Path: {most_recent}")
    
    # Check if aggregate_summary.json exists in the most recent run or outputs
    summary_path = most_recent / "aggregate_summary.json"
    if not summary_path.exists():
        # Check other runs
        print("  [-] aggregate_summary.json not found in the most recent run.")
        # Try to find any aggregate_summary.json
        all_summaries = sorted(list(runs_dir.glob("**/aggregate_summary.json")), key=os.path.getmtime, reverse=True)
        if all_summaries:
            summary_path = all_summaries[0]
            print(f"  [+] Found an aggregate summary in a different run: {summary_path}")
        else:
            summary_path = None
            
    if summary_path and summary_path.exists():
        # Fail loud on a corrupt summary instead of degrading to the misleading
        # "Run the paper-memory suite" recommendation. A parse failure here means
        # the suite already RAN and its ``aggregate_summary.json`` artifact is
        # DAMAGED, so re-running the expensive suite (the prior None -> "run
        # again" path) burns GPU without addressing the corruption — and leaves
        # corruption indistinguishable from "no summary yet". Surface the file +
        # cause on stderr and exit non-zero so an operator (or the Makefile
        # status target) knows the check failed on a CORRUPT artifact, not a
        # missing one. Same fail-loud posture as scripts/best_run_reader.py et
        # al.; the recovery instruction's "Exception丸呑み -> propagate" rule
        # (the old broad ``except Exception`` also masked e.g. a PermissionError
        # as a parse failure).
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            sys.stderr.write(
                f"[-] {summary_path} is corrupt: not valid JSON "
                f"({exc.msg} at line {exc.lineno} column {exc.colno}). The "
                f"summary is DAMAGED, not missing — investigate the file "
                f"instead of re-running the paper-memory suite.\n"
            )
            raise SystemExit(2)
        except OSError as exc:
            sys.stderr.write(f"[-] {summary_path} unreadable ({exc}).\n")
            raise SystemExit(2)
        print(f"[+] Loaded {summary_path.name}")
        return data

    return None

def load_halt_guard_module():
    """Load ``scripts/loop_halt_guard.py`` as a module, by file location.

    The status check runs as a bare CLI (``make check-status`` →
    ``python scripts/agent_check_status.py``), so the sibling guard is not
    importable as a package member from every invocation context — load it by
    path, the same idiom ``tests/test_agent_check_status_gpu.py`` uses to load
    THIS script. Returns ``None`` when the guard is absent/unloadable: the halt
    state is then simply not consultable, and this diagnostic keeps its legacy
    recommendation flow rather than failing (a missing sibling must not brick
    the status check)."""
    guard_path = Path(__file__).resolve().parent / "loop_halt_guard.py"
    try:
        spec = importlib.util.spec_from_file_location("loop_halt_guard", guard_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        # Register BEFORE exec — the manual-loading recipe — so module-scoped
        # machinery that resolves through sys.modules (the guard's frozen
        # dataclass) finds the module instead of crashing on None.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except (OSError, ImportError, SyntaxError):
        return None
    return module


def loop_halt_decision():
    """The loop-halt guard's verdict for the §4 / MS-008 axis, or ``None`` when
    not consultable.

    Resolved from the CURRENT WORKING DIRECTORY — the same convention as
    ``check_datasets``' relative ``data`` globs and the guard CLI's own
    ``--repo-root .`` default, so ``make check-status`` (always run from the
    repo root) consults the repo's live ``loop_axis_state.json``. A corrupt
    tracker (``JSONDecodeError`` subclasses ``ValueError``) degrades to ``None``
    — a broken halt tracker must not brick the status check either."""
    guard = load_halt_guard_module()
    if guard is None:
        return None
    try:
        return guard.should_skip(Path.cwd())
    except (OSError, ValueError):
        return None


def halt_operator_decisions() -> list[str]:
    """The operator's unblock set when the loop-halt guard says SKIP.

    SKIP does not mean "nothing can be done" — it means the AGENT must produce
    nothing while awaiting one of the three witnessed external triggers (see
    ``loop_axis_state.json``). Surfacing them here keeps ``make check-status``
    (the command the operator already runs) from routing anyone back into
    ``make prepare-data`` / ``make paper-memory`` work the halt has suspended."""
    return [
        "Operator decision (halt is ACTIVE — one of the 3 triggers unblocks it):",
        "  (A) Fire a real operator-side 9B run -> a new tests/fixtures/"
        "freeze_validloss_ci_9b*.json",
        "      deposit appears (count > baseline in loop_axis_state.json).",
        "  (B) Draft + approve the MS-008 publishable-negative closeout ->",
        "      reports/close-the-loop/close_the_loop_funnel_go_nogo_*.{json,md} appears.",
        "  (C) Open a new measurable axis -> operator_signals.new_ms_axis_opened is set",
        "      in loop_axis_state.json (e.g. MS-009).",
    ]


def halt_proceed_minimal_steps(active_triggers) -> list[str]:
    """The minimal step for each FIRED trigger, in trigger order.

    On PROCEED the loop's contract narrows to the fired axis only (PURPOSE.md:
    "triggers が示す該当軸の最小 step のみ実行する") — so the surfaced step must
    name that axis's next action, never the generic milestone menu."""
    steps = {
        "9b_leg_fired": [
            "  [9b_leg_fired] New 9B deposit witnessed -> run the Cat-C minimal step:",
            "    compare target-scale 9B absolute valid_loss against the",
            "    PRODUCTION baseline (the one remaining §4 open).",
        ],
        "closeout_approved": [
            "  [closeout_approved] Closeout anchor witnessed -> finalize the",
            "    MS-008 publishable-negative closeout from the approved anchor.",
        ],
        "new_ms_axis_opened": [
            "  [new_ms_axis_opened] New axis signal witnessed -> read",
            "    loop_axis_state.json's triggers and run ONLY the new axis's",
            "    minimal step (e.g. the MS-009 first step).",
        ],
    }
    lines: list[str] = []
    for trigger in active_triggers:
        lines.extend(
            steps.get(
                trigger,
                [
                    f"  [{trigger}] Unknown trigger -> consult loop_axis_state.json",
                    "    and run that axis's minimal step only.",
                ],
            )
        )
    return lines


def evaluate_and_suggest(data_ok, summary_data):
    """Print the milestone/next-step evaluation and return the guard's verdict.

    The return value is the loop-halt guard's ``SkipDecision`` — skip or
    PROCEED — or ``None`` when the guard/tracker is not consultable. The caller
    (``main``) needs the verdict to gate the halt-inconsistent trailing stage:
    under SKIP the 9B readiness block would hand the operator a second
    "advance the lever" decision menu right after the "produce nothing" one."""
    print("\n=== [3/3] Milestone & Next Step Evaluation ===")

    halt = loop_halt_decision()
    if halt is not None and halt.skip:
        # awaiting_ratification with no witnessed trigger: every command this
        # stage normally recommends (make prepare-data / make paper-memory /
        # make paper-memory-evaluate-gates) is axis work the halt suspends.
        # Surface the verdict + the operator unblock set INSTEAD — the
        # auto-diagnostic must not send anyone back into blocked work.
        print("[!] Loop halt guard: SKIP (halt — produce nothing)")
        print(f"    status:          {halt.status}")
        print(f"    active_triggers: {halt.active_triggers or '(none)'}")
        print(f"    reason:          {halt.reason}")
        print("Recommendation:")
        print("  -> Produce NOTHING this iteration. The usual next steps")
        print("     (make prepare-data / make paper-memory / make")
        print("     paper-memory-evaluate-gates) are BLOCKED by the halt.")
        print("    Confirm the verdict with: make loop-halt-check")
        for _decision_line in halt_operator_decisions():
            print(_decision_line)
        return halt

    if halt is not None and not halt.skip and halt.active_triggers:
        # PROCEED via a WITNESSED trigger: the contract narrows to the fired
        # axis's minimal step only. Falling through to the generic
        # prepare-data / paper-memory menu here is the D-2 misroute — the
        # trigger fired, but the operator (or a goaldev agent) still cannot
        # tell WHICH axis's minimal step to run.
        print("[!] Loop halt guard: PROCEED (trigger witnessed)")
        print(f"    status:          {halt.status}")
        print(f"    active_triggers: {halt.active_triggers}")
        print("Minimal step(s) for the fired axis ONLY:")
        for _step_line in halt_proceed_minimal_steps(halt.active_triggers):
            print(_step_line)
        print("    Confirm the verdict with: make loop-halt-check")
        return halt

    if not data_ok:
        print("Recommendation:")
        print("  -> Run data preparation to set up the 5K Dolly dataset split.")
        print("  Command: make prepare-data")
        return halt

    if not summary_data:
        print("Recommendation:")
        print("  -> Run the 3-seed paper-memory suite to perform the core experiments.")
        print("  Command: make paper-memory")
        return halt
        
    # Analyze summary data
    print("Current Experiment Results Summary:")
    print(f"  Seeds evaluated: {summary_data.get('seeds', 'Unknown')}")
    
    # We will try to read the paper gates report if generated, or evaluate G1-G4
    print("\nNext Steps:")
    print("  -> Review the aggregate summary and run gate evaluation:")
    print("  Command: make paper-memory-evaluate-gates")
    
    # If G3 is not run, run external quality evaluation
    print("  -> Run external evaluation (ARC, HellaSwag, etc.) to pass G3:")
    print("  Command: make paper-memory-external-eval")
    return halt

def parse_gpu_holders(apps_csv):
    """Parse ``nvidia-smi --query-compute-apps=pid,process_name,used_memory``
    CSV output into a list of ``{pid, name, mem}`` dicts, one per process
    holding GPU compute. An empty/whitespace result (no compute apps) yields
    ``[]`` — i.e. the GPU looks free. Pure: takes the CSV text so it is testable
    without a real GPU. Tolerant of malformed rows (too few columns → skipped),
    since nvidia-smi's CSV is the only input and a half-parsed line must not
    brick the status check."""

    holders = []
    for line in apps_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        # pid, process_name, used_memory — name itself may contain commas on
        # some drivers, so keep everything from field 1 onward as the name and
        # treat the final field as the memory figure when there are extras.
        if len(parts) < 3:
            continue
        holders.append({"pid": parts[0], "name": parts[1], "mem": parts[-1]})
    return holders


def query_gpu_compute_apps():
    """Return ``nvidia-smi --query-compute-apps`` stdout (CSV) or ``None`` when
    nvidia-smi is absent, times out, or exits non-zero. GPU assessment is
    INFORMATIONAL — a host with no GPU (CI, CPU-only dev box, MLX Track-B) must
    still pass ``make status`` — so any failure to probe degrades to "cannot
    assess", never to a status-check failure."""

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def src_data_pipeline_present() -> bool:
    """Whether the private ``src.data`` data pipeline ships in this checkout.

    The 9B target-scale run (GOAL §4) depends on ``src.data`` to produce/serve
    its 9B samples. On the PUBLIC mirror that pipeline is deliberately stripped
    (DATA/Cat-C — see PURPOSE.md), so the 9B lever is NOT actionable from this
    checkout regardless of GPU state; on the PRIVATE checkout it ships and the
    run's real readiness gate is GPU availability. Resolved from ``__file__``
    (not CWD) so the answer is stable however ``make status`` is invoked, and a
    module-level function so tests can inject both contexts without a GPU."""
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "src" / "data").is_dir()


def data_blocked_operator_decision() -> list[str]:
    """The operator's forward decision set when the 9B lever is DATA-blocked.

    The status report diagnoses the block (``src.data`` stripped, DATA/Cat-C);
    this completes the *decision* by stating what the operator can actually DO
    — the three forward options that advance the loop off its terminal state.
    Surfacing them here (inside ``make status``, the command the operator
    already runs) is the disciplined alternative to the loop's recurring
    peripheral-plumbing churn: the block got re-derived every iteration because
    the DECISION behind it was never printed in the tool the operator reads.
    """
    return [
        "Operator decision (the loop's terminal state here — pick one to advance):",
        "  (A) Accept SHIP as final for this public mirror — the §4 arc is complete",
        "      (both citable faithful TIES + quality-vs-full-backprop SURPASSES, harvested).",
        "  (B) Run the private-src.data absolute-loss leg in the PRIVATE checkout, where",
        "      src.data ships — the one remaining open scientific question (DATA/Cat-C).",
        "  (C) Define a NEW Cat-A deliverable to pivot the loop off the exhausted §4 arc.",
    ]


def report_gpu_availability():
    """Surface what concretely blocks the 9B target-scale lever THIS cycle.

    This is the recurring question the auto-loop kept answering with another
    report-tooling hardening pass instead of resolving: "is the GPU available,
    and does freeing it actually unblock the 9B run?". The lever has TWO
    distinct blocks, only one of which is GPU — and conflating them (the prior
    copy always said "stop the holder to make the GPU-bound 9B run actionable")
    sent the operator to free a GPU that, on this public mirror, cannot move the
    lever:

    * DATA-blocked (this public mirror): ``src.data`` is stripped (DATA/Cat-C),
      so freeing the GPU does NOT make the lever actionable here — the §4
      verdict runs are already harvested as TIES and the only remaining open is
      private-``src.data`` absolute loss. Naming this stops the operator burning
      a cycle on a GPU free-up that cannot move the lever.
    * GPU-blocked (private checkout, ``src.data`` present): name the holder
      process (e.g. a sibling project's ``llama-server``) to turn an opaque
      "GPU busy" into a concrete unblock ("stop PID X").

    Informational only: any outcome writes to stdout and returns normally — a
    busy/absent GPU or a stripped pipeline never fails the status check; it only
    changes which next step is actionable."""

    print("\n=== 9B Lever Readiness (next-lever block) ===")
    data_present = src_data_pipeline_present()
    if not data_present:
        # The public mirror: the lever's block is the missing data pipeline, NOT
        # the GPU. State this up front so the GPU lines below read as the
        # informational context they are, not as the lever's gate.
        print("[!] 9B lever is DATA-blocked, not GPU-blocked.")
        print("    private src.data is stripped from this public mirror (DATA/Cat-C);")
        print("    freeing the GPU will NOT make the 9B run actionable here. The §4")
        print("    verdict runs are already harvested as TIES; the remaining open is")
        print("    private-src.data absolute loss (actionable only in the private checkout).")
        for _decision_line in data_blocked_operator_decision():
            print(_decision_line)

    apps_csv = query_gpu_compute_apps()
    if apps_csv is None:
        print("[ ] nvidia-smi not available — cannot assess GPU (CPU-only / no-NVIDIA host).")
        return
    holders = parse_gpu_holders(apps_csv)
    if not holders:
        print("[+] No compute apps on the GPU — it appears FREE.")
        if data_present:
            print("    -> GPU free AND src.data present: the 9B lever is actionable this cycle.")
        else:
            print("    -> But the lever stays DATA-blocked here — not actionable from this mirror.")
        return
    print(f"[!] GPU is held by {len(holders)} compute app(s):")
    for h in holders:
        print(f"    PID {h['pid']:<10} {h['mem']:<10} {h['name']}")
    if data_present:
        print("    -> Stop the holder(s) above to make the GPU-bound 9B run actionable.")
    else:
        print("    -> Freeing this GPU will NOT move the lever — it is DATA-blocked (see above).")


def main():
    data_ok = check_datasets()
    summary_data = check_experiment_runs()
    halt = evaluate_and_suggest(data_ok, summary_data)
    # SKIP verdict: stage 3 already printed the operator's halt-unblock set, and
    # the 9B readiness block would append a second, halt-inconsistent
    # "advance the lever" decision menu — suppress it for the same reason the
    # stage-3 recommendations are supplanted. Guard not consultable (None) or
    # PROCEED: the readiness block keeps its legacy unconditional behavior.
    if halt is None or not halt.skip:
        report_gpu_availability()

if __name__ == "__main__":
    main()
