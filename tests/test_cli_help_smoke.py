"""CLI smoke tests: verify every argparse entry-point responds to --help.

The parametrization is **auto-discovered**, not hand-picked: every
``scripts/*.py`` with an ``if __name__ == "__main__"`` guard is swept, split by
whether it constructs an :class:`argparse.ArgumentParser` (``--help`` must exit 0)
or uses raw ``sys.argv`` (import-only — its top-level imports must resolve when
invoked the way users invoke it). Adding a new CLI script needs no edit here; it
is covered the moment it lands under ``scripts/``. See :func:`_discover`.

This suite is also the **bootstrap-defect canary**: it fails the moment a
script can no longer find the repo root and thus cannot import an in-repo
``src.*`` / ``scripts.*`` module (the ``sys.path.insert(repo_root)`` idiom every
standalone CLI must carry). To keep that signal honest, each script is launched
as a subprocess with ``PYTHONPATH`` **stripped** (see :data:`_SUBPROCESS_ENV`)
so it runs under a clean path — as in real CI — rather than inheriting this AI
Hub worktree's ``PYTHONPATH=/home/jinno/ai-hub`` (whose regular
``scripts/__init__.py`` package shadows the repo's namespace ``scripts/``, per
PEP 420, and would mask every ``from scripts.X import`` as a false failure).
A residual ``--help`` failure is then *classified* — only a genuine in-repo
import break (or a missing optional dependency, or the private ``src.data``
pipeline stripped from this public mirror) is surfaced; see
:func:`_classify_cli_help_failure`.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Subprocess environment with ``PYTHONPATH`` stripped. The AI Hub shell sets
# ``PYTHONPATH=/home/jinno/ai-hub``; its regular ``scripts/__init__.py`` package
# shadows the repo's namespace ``scripts/`` (PEP 420: a regular package found
# *anywhere* on sys.path wins over a namespace package found *earlier*), so a
# subprocess that inherits it reports a false ``No module named 'scripts.X'`` for
# every ``from scripts.X import`` — masking scripts that work perfectly under CI.
# Stripping PYTHONPATH makes the subprocess's bootstrap put the repo root (and
# thus the repo's own ``scripts/``) on sys.path uncontested, so the canary reads
# the script's true CI behavior. (Real CI has no ``/home/jinno/ai-hub`` on path.)
_SUBPROCESS_ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
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
# A module-load failure is the only non-zero-exit shape the IMPORT_ONLY arm treats
# as a bootstrap signal: ``ModuleNotFoundError`` / ``ImportError`` / ``SyntaxError``
# raised while importing the script. Any other non-zero exit (the script's own
# ``Usage:`` exit on a missing argument, an argparse error, a runtime exception in
# ``main``) means the top-level imports already resolved — a pass, not a defect.
_MODULE_LOAD_FAILURE = re.compile(r"ModuleNotFoundError|ImportError|SyntaxError")


# ---------------------------------------------------------------------------
# Auto-discovery: every __main__-guarded scripts/*.py entry point.
# ---------------------------------------------------------------------------


def _has_main_guard(tree: ast.AST) -> bool:
    """True iff the module has an ``if __name__ == "__main__":`` guard."""
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                return True
    return False


def _has_argparse_entry_point(tree: ast.AST) -> bool:
    """True iff the module constructs an :class:`argparse.ArgumentParser`.

    Covers both ``argparse.ArgumentParser()`` and a bare ``ArgumentParser()``
    (after ``from argparse import ArgumentParser``), anywhere in the module —
    including a parser built lazily inside ``main()``.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
                return True
            if isinstance(func, ast.Name) and func.id == "ArgumentParser":
                return True
    return False


def _imports_scripts_package(tree: ast.AST) -> bool:
    """True iff the module has a top-level import of the ``scripts`` package.

    ``from scripts.X import ...`` / ``import scripts.X``. These resolve only
    when the repo root (the parent of ``scripts/``) is on ``sys.path`` — which a
    bare ``python scripts/foo.py`` invocation does NOT provide (it puts
    ``scripts/`` itself on the path). A script with such an import MUST carry
    the repo-root bootstrap, or it is broken under direct invocation.
    """
    for node in tree.body:  # top-level statements only
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "scripts" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "scripts":
                return True
    return False


def _has_repo_root_bootstrap(source: str) -> bool:
    """True iff the source puts the repo root on ``sys.path``.

    Recognizes the whole family of equivalent idioms in the tree — the inline
    ``sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`` form, the
    ``append`` variant, and the named-constant form
    ``repo_root = Path(__file__).resolve().parents[1]; sys.path.append(...)``.
    The invariant signal across all: a ``sys.path.insert``/``append`` call
    together with a ``parents[1]`` repo-root computation from ``__file__`` (the
    repo root is one directory above ``scripts/``). The guard polices the
    *presence* of a bootstrap, not its exact form, so it does not churn
    pre-existing equivalent idioms.
    """
    has_path_op = ("sys.path.insert" in source) or ("sys.path.append" in source)
    return has_path_op and ("parents[1]" in source) and ("__file__" in source)


def _discover():
    """Return ``(argparse_scripts, import_only_scripts)`` under ``scripts/``.

    A script with a ``__main__`` guard is an entry point. If it builds an
    ``ArgumentParser`` it must answer ``--help``; otherwise it is invoked with
    raw ``sys.argv`` and is checked import-only. Files without a ``__main__``
    guard (pure library modules such as ``git_utils.py``) are not entry points
    and are skipped.
    """
    argparse_scripts: list[str] = []
    import_only: list[str] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            # A script that won't even parse is its own defect; surface it via
            # the argparse arm (which will fail loud with the SyntaxError).
            argparse_scripts.append(str(path.relative_to(ROOT)))
            continue
        if not _has_main_guard(tree):
            continue
        rel = str(path.relative_to(ROOT))
        (argparse_scripts if _has_argparse_entry_point(tree) else import_only).append(rel)
    return argparse_scripts, import_only


ARGPARSE_SCRIPTS, IMPORT_ONLY_SCRIPTS = _discover()


# ---------------------------------------------------------------------------
# Failure classification.
# ---------------------------------------------------------------------------


def _inrepo_module_exists(module: str) -> bool:
    """True iff ``module`` (dotted) resolves to a file/package in this checkout.

    ``src.tg_lora.freeze_cost`` → ``src/tg_lora/freeze_cost.py``;
    ``src.model`` → ``src/model/__init__.py`` (a package);
    ``scripts.compare_runs`` → ``scripts/compare_runs.py``.
    """
    rel = Path(*module.split("."))
    return (ROOT / rel.with_suffix(".py")).exists() or (ROOT / rel / "__init__.py").exists()


def _classify_cli_help_failure(stderr: str) -> str:
    """Classify the root cause of a failed ``script --help``.

    The subprocess is run with ``PYTHONPATH`` stripped (see
    :data:`_SUBPROCESS_ENV`), so the AI Hub worktree's regular-package shadow of
    ``scripts`` cannot arise — a ``scripts.*`` import resolves to this repo's own
    namespace ``scripts/``.

    Returns one of:

    - ``"bootstrap_defect"``: a genuine in-repo import break — a ``src.*`` module
      whose leaf EXISTS in this checkout (the bootstrap should resolve ``src`` to
      this repo), ANY ``scripts.*`` ``ModuleNotFoundError`` (under a stripped
      PYTHONPATH that means a missing bootstrap or a missing sibling — both real
      defects), or a non-``No module named`` failure (``SyntaxError`` /
      ``ImportError: cannot import name``). This is the actionable signal the
      canary surfaces — it FAILs.
    - ``"known_unavailable"``: the script failed *only* because an external
      dependency (``peft``/``torch``/...) or the private ``src.data`` pipeline
      (stripped from this public mirror) is absent. Not a regression; the caller
      ``xfail``s.

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
    if top_level == "src":
        # ``src.X``: if the leaf EXISTS in this repo, the script is missing the
        # bootstrap that resolves ``src`` to this checkout — a real, fixable
        # defect. If the leaf does NOT exist here, it is a genuinely
        # missing/private submodule (not a regression).
        return "bootstrap_defect" if _inrepo_module_exists(module) else "known_unavailable"
    if top_level == "scripts":
        # Under the stripped PYTHONPATH ``scripts`` resolves to this repo's
        # namespace ``scripts/`` (the AI Hub regular-package shadow is gone), so a
        # ``scripts.X`` ``ModuleNotFoundError`` is a real defect — the script
        # either lacks the repo-root bootstrap (``scripts`` not importable) or the
        # sibling genuinely does not exist. (The static guard in TestDiscovery
        # catches the missing-bootstrap case at the source.)
        return "bootstrap_defect"
    # External dependency (peft / torch / transformers / datasets / ...).
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
        a genuine bootstrap defect (an in-repo import break that a repo-root
        bootstrap would fix), not on a missing optional dependency, the private
        ``src.data`` pipeline, or this venv's editable ``scripts`` shadow.
        """
        r = subprocess.run(
            [sys.executable, argparse_script_path, "--help"],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
            cwd=str(ROOT),
            env=_SUBPROCESS_ENV,
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
        """Non-argparse (raw ``sys.argv``) script: its top-level imports must
        resolve when invoked the way users invoke it.

        A bare ``python scripts/X.py`` puts the script's own directory — NOT the
        repo root — on ``sys.path``, so an in-repo ``scripts.*`` / ``src.*``
        import that resolves only via a repo-root bootstrap breaks at import
        time. Run the script as a real subprocess and route any module-load
        failure through :func:`_classify_cli_help_failure`, so a genuine bootstrap
        defect fails loud rather than silently passing.

        A non-zero exit that is NOT a module-load failure (the script's own
        ``Usage:`` exit with no argument) means the imports resolved → pass.
        """
        r = subprocess.run(
            [sys.executable, import_script_path],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
            cwd=str(ROOT),
            env=_SUBPROCESS_ENV,
        )
        if r.returncode == 0:
            return
        if not _MODULE_LOAD_FAILURE.search(r.stderr):
            # Imports resolved; the script exited non-zero for its own reasons
            # (missing argument, usage message, ...) — not a bootstrap defect.
            return
        cause = _classify_cli_help_failure(r.stderr)
        if cause == "known_unavailable":
            pytest.xfail(
                f"{import_script_path} blocked by an unavailable dependency/"
                f"private module — not a bootstrap regression"
            )
        pytest.fail(
            f"{import_script_path} BOOTSTRAP DEFECT (classified={cause}):\n{r.stderr[:500]}"
        )


class TestDiscovery:
    """Pin the auto-discovery so the parametrization can't silently rot.

    The contract: every ``__main__``-guarded ``scripts/*.py`` is swept, split by
    argparse presence; library modules (no ``__main__`` guard) are skipped.
    """

    def test_covers_every_main_guarded_script(self):
        main_guarded = {
            p.name
            for p in (ROOT / "scripts").glob("*.py")
            if _has_main_guard(ast.parse(p.read_text()))
        }
        swept = {Path(s).name for s in ARGPARSE_SCRIPTS} | {
            Path(s).name for s in IMPORT_ONLY_SCRIPTS
        }
        assert swept == main_guarded, (
            f"discovery drift — swept != main-guarded: "
            f"missing={main_guarded - swept} extra={swept - main_guarded}"
        )

    def test_known_library_module_is_skipped(self):
        # git_utils.py has no __main__ guard (pure library) → not an entry point.
        assert "scripts/git_utils.py" not in ARGPARSE_SCRIPTS + IMPORT_ONLY_SCRIPTS

    def test_import_only_arm_is_nonempty_and_excludes_argparse_scripts(self):
        # The raw-sys.argv arm must stay populated (analyze_accel_sweep.py is the
        # canonical member) and must not overlap the argparse arm.
        assert IMPORT_ONLY_SCRIPTS, "import-only arm unexpectedly empty"
        assert not (set(ARGPARSE_SCRIPTS) & set(IMPORT_ONLY_SCRIPTS))

    def test_scripts_package_import_requires_repo_root_bootstrap(self):
        """Every script importing from the ``scripts`` package carries the bootstrap.

        ``from scripts.X import ...`` resolves only via the repo root on
        ``sys.path``; a bare ``python scripts/foo.py`` puts ``scripts/`` (not its
        parent) on the path, so the import breaks under direct invocation
        without the ``sys.path.insert(repo_root)`` idiom. The runtime canary's
        subprocess strips PYTHONPATH so ``scripts`` resolves to this repo, which
        makes a missing-bootstrap ``scripts.*`` import surface as a defect — but
        only if that script is actually swept. This STATIC guard is the
        belt-and-suspenders check that catches the defect at the source even for
        a script not yet wired into a parametrized arm, and documents the
        invariant every ``scripts.*`` importer must satisfy.
        """
        offenders = []
        for path in (ROOT / "scripts").glob("*.py"):
            source = path.read_text()
            tree = ast.parse(source)
            if _imports_scripts_package(tree) and not _has_repo_root_bootstrap(source):
                offenders.append(path.name)
        assert not offenders, (
            "scripts importing from the `scripts` package lack the repo-root "
            "bootstrap (would break under direct invocation / be masked as a "
            f"venv shadow by the runtime canary): {offenders}"
        )


class TestClassifyCliHelpFailure:
    """Pin the canary's failure discrimination so it can't silently rot.

    The contract: a genuine in-repo import break (``src.*`` leaf that exists
    here; any ``scripts.*`` failure under the stripped-PYTHONPATH subprocess; a
    non-``No module named`` failure) is a ``bootstrap_defect`` (the suite must
    fail); only a missing optional dependency or the private ``src.data``
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
            # ``scripts.*`` whose sibling EXISTS in this repo: the canary runs the
            # subprocess with PYTHONPATH stripped, so ``scripts`` resolves to this
            # repo's namespace ``scripts/`` — a failure here means the script lacks
            # the repo-root bootstrap (``scripts`` not importable) → real defect.
            "No module named 'scripts.compare_runs'",
            "No module named 'scripts.run_paper_external_eval'",
            "No module named 'scripts.replay_freeze_validloss_ci'",
            # ``scripts.*`` whose sibling does NOT exist: a genuinely broken
            # import → defect.
            "No module named 'scripts.totally_missing'",
        ],
        ids=["compare_runs-exists", "run_paper_external_eval-exists",
             "replay_freeze_validloss_ci-exists", "sibling-missing"],
    )
    def test_scripts_import_failure_is_defect(self, stderr):
        # Under the stripped-PYTHONPATH subprocess, there is no scripts shadow to
        # hide behind: a ``scripts.*`` ModuleNotFoundError is always a real defect
        # (missing bootstrap or missing sibling). The static guard in TestDiscovery
        # is the primary check for the missing-bootstrap case; this pins the
        # runtime classifier so it never masks a scripts.* failure as xfail.
        assert _classify_cli_help_failure(stderr) == "bootstrap_defect"

    @pytest.mark.parametrize(
        "stderr",
        [
            # A ``src.*`` whose module EXISTS in this repo (or the bare top-level
            # ``src``): the script is missing the repo-root bootstrap that resolves
            # ``src`` to this checkout (the editable leak otherwise hands it a stale
            # private ``src``). A real, fixable defect — this is the canary's signal.
            "No module named 'src'",  # repo root not on sys.path at all
            "No module named 'src.model'",  # repo root importable, package lost
            "No module named 'src.tg_lora.freeze_cost'",  # leaf module
            "No module named 'src.model.load_model'",  # deeper leaf
            "No module named 'src.utils.device'",
            "No module named 'src.tg_lora.prefix_feature_cache'",
        ],
        ids=["missing-root-src", "src.model", "freeze_cost", "load_model", "device", "prefix_feature_cache"],
    )
    def test_existing_src_leaf_is_bootstrap_defect(self, stderr):
        assert _classify_cli_help_failure(stderr) == "bootstrap_defect"

    def test_missing_src_leaf_is_known_unavailable(self):
        # A src.* import whose leaf does NOT exist in this repo (genuinely
        # missing / private submodule not mapped by the pipeline regex) is not a
        # bootstrap regression.
        assert (
            _classify_cli_help_failure("No module named 'src.tg_lora.no_such_leaf'")
            == "known_unavailable"
        )

    def test_downstream_attribution_does_not_mask_real_inrepo_defect(self):
        # The discriminating case for the private-pipeline attribution: a
        # traceback whose final error names a src.* leaf that EXISTS here (so it
        # WOULD be a defect bare), carried in a chain that does NOT run through
        # ``src.data``. The :func:`_failure_originates_from_private_pipeline`
        # regex must NOT match (no ``src.data`` frame), so this stays a defect —
        # proving the downstream attribution is root-cause-based, not masking.
        stderr = (
            "Traceback (most recent call last):\n"
            "  File \"scripts/foo.py\", line 3, in <module>\n"
            "    from src.model.lora_utils import iter_lora_params\n"
            "ModuleNotFoundError: No module named 'src.model'"
        )
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
