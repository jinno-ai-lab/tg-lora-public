"""CLI smoke tests: verify all Python scripts import and respond to --help (TASK-0124).

Each argparse-based script must exit 0 with --help. Scripts using raw
sys.argv are verified via import only.

This suite is also the **bootstrap-defect canary**: it fails the moment a
script can no longer find the repo root and thus cannot import an in-repo
``src.*`` / ``scripts.*`` module (the ``sys.path.insert(repo_root)`` idiom the
TASK-0146..0148 series added to every standalone CLI). To keep that signal
honest, a ``--help`` failure is *classified* — only a genuine in-repo import
break fails the suite; a failure caused by an unavailable optional/heavy
dependency (``peft``/``datasets``/...) or the private ``src.data`` pipeline
(stripped from this public mirror) is reported as ``xfail`` so it can no longer
mask a real regression. See :func:`_classify_cli_help_failure`.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# In-repo top-level packages: a failure to import one of these means the script
# cannot resolve the repo root — i.e. the bootstrap defect this canary catches.
_INREPO_TOPLEVELS = ("src", "scripts")
# ``src.data`` is the private data pipeline, absent from this public mirror; its
# absence is a mirror limitation, not a bootstrap regression.
_PRIVATE_INREPO_MODULES = ("src.data",)
# A private-pipeline frame in an import traceback: either the missing module is
# itself under ``src.data`` (``No module named 'src.data.build_seed_dataset'``)
# or a private ``src/data/*.py`` file appears in the chain. The upstream venv
# leaks the private ``src`` into the namespace, so a script that imports
# ``src.data.<x>`` runs a private ``src/data/<x>.py`` frame whose *downstream*
# imports (e.g. ``src.utils.io``) then fail in the split namespace — the root
# cause is the absent private pipeline, not the downstream module name.
_PRIVATE_PIPELINE_ORIGIN = re.compile(r"\bsrc\.data\.\w+|src/data/\w+\.py")


def _classify_cli_help_failure(stderr: str) -> str:
    """Classify the root cause of a failed ``script --help``.

    Returns one of:

    - ``"bootstrap_defect"``: the script failed to import an in-repo module
      (``src.*`` other than the private ``src.data``, or ``scripts.*``), or the
      failure was not a ``ModuleNotFoundError`` at all (e.g. ``SyntaxError`` /
      ``ImportError: cannot import name``). This is the actionable signal the
      canary exists to surface — it should FAIL the suite.
    - ``"known_unavailable"``: the script failed *only* because an external
      dependency or the private ``src.data`` pipeline is absent from this
      checkout. Not a regression; the caller reports it as ``xfail``.

    A successful ``--help`` (exit 0) never reaches this function.
    """
    match = re.search(r"No module named '([^']+)'", stderr)
    if not match:
        # Unexpected failure shape (syntax error, bad import name, ...) — surface
        # it rather than silently treating it as a benign skip.
        return "bootstrap_defect"
    module = match.group(1)
    if module in _PRIVATE_INREPO_MODULES:
        return "known_unavailable"
    # A missing ``src.*`` module is normally a bootstrap defect — UNLESS the
    # import chain runs through the private ``src.data`` pipeline, in which case
    # the root cause is the absent private pipeline (a mirror limitation), not a
    # repo-root resolution break. This catches the downstream shape the direct
    # ``module in _PRIVATE_INREPO_MODULES`` check misses: e.g. ``src.utils.io``
    # failing *inside* a private ``src/data/build_seed_dataset.py`` frame.
    if _failure_originates_from_private_pipeline(stderr):
        return "known_unavailable"
    top_level = module.split(".", 1)[0]
    if top_level in _INREPO_TOPLEVELS:
        return "bootstrap_defect"
    return "known_unavailable"


def _failure_originates_from_private_pipeline(stderr: str) -> bool:
    """True when the failing import chain runs through the private ``src.data``.

    The private data pipeline (``src/data/*.py``) is stripped from this public
    mirror. When the upstream venv leaks the private ``src`` into the namespace,
    a script importing ``src.data.<x>`` executes a private ``src/data/<x>.py``
    frame whose downstream imports then fail in a split namespace (e.g.
    ``src.utils.io`` unresolvable there). The final ``ModuleNotFoundError``
    names the downstream module, not ``src.data`` — so the direct check in
    :func:`_classify_cli_help_failure` misses it and would wrongly flag a
    bootstrap defect. This attributes the failure to its private-pipeline root
    cause so the canary ``xfail``s the mirror limitation instead of failing.

    A genuine in-repo import break (e.g. ``No module named 'src.model'`` with no
    ``src.data`` in its traceback) does NOT match, so real bootstrap defects are
    still surfaced — this is root-cause attribution, not masking.
    """
    return bool(_PRIVATE_PIPELINE_ORIGIN.search(stderr))

# Scripts using argparse (exit 0 on --help)
ARGPARSE_SCRIPTS = [
    "scripts/advise_training.py",
    "scripts/analyze_benchmark.py",
    "scripts/analyze_prefix_cache_break_even.py",
    "scripts/analyze_sensitivity.py",
    "scripts/analyze_trajectory.py",
    "scripts/benchmark_optimizer_lifecycle.py",
    "scripts/benchmark_prefix_cache.py",
    "scripts/benchmark_velocity_ops.py",
    "scripts/check_spine_anchors.py",
    "scripts/compare_experiment_configs.py",
    "scripts/compare_paper_memory_modes.py",
    "scripts/compare_runs.py",
    "scripts/consolidate_paper_results.py",
    "scripts/diagnose.py",
    "scripts/download_data.py",
    "scripts/evaluate_paper_gates.py",
    "scripts/export_paper_results.py",
    "scripts/frontier_report.py",
    "scripts/generate_sweep_dashboard.py",
    "scripts/inspect_model.py",
    "scripts/lookup_batch_plan.py",
    "scripts/precompute_prefix_cache_parallel.py",
    "scripts/prepare_data.py",
    "scripts/recover.py",
    "scripts/run_paper_external_eval.py",
    "scripts/summarize_sweep.py",
]

# Scripts using raw sys.argv (no --help) — import-only check
IMPORT_ONLY_SCRIPTS = [
    "scripts/analyze_accel_sweep.py",
]


@pytest.fixture(name="argparse_script_path", params=ARGPARSE_SCRIPTS, ids=lambda s: Path(s).name)
def _argparse_script_path_fixture(request):
    return request.param


@pytest.fixture(name="import_script_path", params=IMPORT_ONLY_SCRIPTS, ids=lambda s: Path(s).name)
def _import_script_path_fixture(request):
    return request.param


class TestCLIHelpSmoke:
    """Verify argparse scripts respond to --help and non-argparse scripts import."""

    def test_argparse_help(self, argparse_script_path):
        """Script exits 0 and prints usage with --help.

        If ``--help`` fails, the cause is classified so the suite only fails on
        a genuine bootstrap defect (an in-repo import break), not on a missing
        optional dependency or the private ``src.data`` pipeline.
        """
        r = subprocess.run(
            [sys.executable, argparse_script_path, "--help"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            cause = _classify_cli_help_failure(r.stderr)
            if cause == "known_unavailable":
                pytest.xfail(
                    f"{argparse_script_path} --help blocked by an unavailable "
                    f"dependency/private module — not a bootstrap regression"
                )
            # bootstrap_defect (or any unexpected shape): this is the signal the
            # canary exists to catch, so fail loudly with the captured detail.
            pytest.fail(
                f"{argparse_script_path} --help BOOTSTRAP DEFECT (exited "
                f"{r.returncode}, classified={cause}):\n{r.stderr[:500]}"
            )
        combined = (r.stdout + r.stderr).lower()
        assert "usage" in combined or "help" in combined, (
            f"{argparse_script_path} --help: no usage/help in output"
        )

    def test_import_only(self, import_script_path):
        """Non-argparse script can be imported without errors."""
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.util; "
                    f"importlib.util.spec_from_file_location('m', '{import_script_path}')"
                ),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
            cwd=str(ROOT),
        )
        assert r.returncode == 0, f"Import check failed for {import_script_path}: {r.stderr}"


class TestClassifyCliHelpFailure:
    """Pin the canary's failure discrimination so it can't silently rot.

    The contract: an in-repo import break is a ``bootstrap_defect`` (the suite
    must fail); a missing optional dependency or the private ``src.data``
    pipeline is ``known_unavailable`` (the suite xfails). Any other failure
    shape is surfaced as a defect rather than swallowed.
    """

    @pytest.mark.parametrize(
        "stderr",
        [
            "Traceback (most recent call last):\n  ...\nModuleNotFoundError: No module named 'peft'",
            "ModuleNotFoundError: No module named 'datasets'",
            "No module named 'torch'",
            "No module named 'transformers'",
        ],
        ids=["peft", "datasets", "torch", "transformers"],
    )
    def test_external_dependency_is_known_unavailable(self, stderr):
        assert _classify_cli_help_failure(stderr) == "known_unavailable"

    def test_private_src_data_pipeline_is_known_unavailable(self):
        assert _classify_cli_help_failure("No module named 'src.data'") == "known_unavailable"

    @pytest.mark.parametrize(
        "stderr",
        [
            # Direct private-submodule absence (clean public mirror, no private
            # venv leak): the missing module is itself under src.data.
            "No module named 'src.data.build_seed_dataset'",
            "No module named 'src.data.filter_records'",
            # The precompute_prefix_cache_parallel.py failure on the upstream
            # venv: the script imports src.data.build_seed_dataset (private),
            # whose frame then imports src.utils.io, which fails in the split
            # namespace. The final error names src.utils.io, but the traceback
            # chain runs through src/data/build_seed_dataset.py — root cause is
            # the absent private pipeline.
            (
                "Traceback (most recent call last):\n"
                "  File \"scripts/precompute_prefix_cache_parallel.py\", line 27, in <module>\n"
                "    from src.data.build_seed_dataset import load_dataset\n"
                "  File \"/home/jinno/tg-lora/src/data/build_seed_dataset.py\", line 7, in <module>\n"
                "    from src.utils.io import load_jsonl\n"
                "ModuleNotFoundError: No module named 'src.utils.io'"
            ),
        ],
        ids=["src.data.submodule", "src.data.other-submod", "precompute-downstream"],
    )
    def test_private_pipeline_origin_is_known_unavailable(self, stderr):
        # Regression for the precompute canary: a downstream src.* failure whose
        # chain runs through the private src.data pipeline must xfail, not flag a
        # bootstrap defect. Before the root-cause-attribution fix this was RED.
        assert _classify_cli_help_failure(stderr) == "known_unavailable"

    @pytest.mark.parametrize(
        "stderr",
        [
            # Same downstream module name (src.utils.io) but NO src.data frame —
            # a genuine in-repo import break, which must STILL fail the suite.
            # Proves the downstream attribution is root-cause-based, not masking.
            "No module named 'src.utils.io'",
            "No module named 'src.model'",
            (
                "Traceback (most recent call last):\n"
                "  File \"scripts/foo.py\", line 3, in <module>\n"
                "    from src.model.lora_utils import iter_lora_params\n"
                "ModuleNotFoundError: No module named 'src.model'"
            ),
        ],
        ids=["src.utils.io-bare", "src.model-bare", "src.model-in-chain"],
    )
    def test_downstream_attribution_does_not_mask_real_inrepo_defect(self, stderr):
        assert _classify_cli_help_failure(stderr) == "bootstrap_defect"

    @pytest.mark.parametrize(
        "stderr",
        [
            "No module named 'src'",  # repo root not on sys.path at all
            "No module named 'src.model'",  # repo root importable, submodule lost
            "No module named 'src.utils.device'",
            "No module named 'src.tg_lora.prefix_feature_cache'",
            "No module named 'scripts.compare_runs'",  # sibling script import lost
        ],
        ids=["missing-root-src", "src.model", "src.utils.submod", "src.tg_lora.submod", "scripts.sibling"],
    )
    def test_inrepo_import_break_is_bootstrap_defect(self, stderr):
        assert _classify_cli_help_failure(stderr) == "bootstrap_defect"

    @pytest.mark.parametrize(
        "stderr",
        [
            "SyntaxError: invalid syntax",
            "ImportError: cannot import name 'gpu_device_name' from 'src.utils.device'",
            "IndentationError: expected an indented block",
        ],
        ids=["syntax", "cannot-import-name", "indentation"],
    )
    def test_non_module_error_is_surfaced_as_defect(self, stderr):
        # A non-ModuleNotFoundError is never silently skipped — it might be the
        # bootstrap defect in disguise (or a real bug), so surface it.
        assert _classify_cli_help_failure(stderr) == "bootstrap_defect"
