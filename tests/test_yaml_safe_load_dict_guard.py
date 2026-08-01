"""Static guard + behavior tests: every ``yaml.safe_load`` → dict-access is
``isinstance(..., dict)``-guarded — close the non-dict-after-yaml.safe_load crash
class.

This is the yaml sibling of ``tests/test_json_loads_dict_guard.py`` for the ONE
other in-repo deserializer. The crash class is identical in shape:

  data = yaml.safe_load(f)      # empty file → None; non-mapping → list/scalar/str
  data.get("key")               # AttributeError: 'NoneType'/'list' has no .get

``yaml.safe_load`` does NOT raise on an empty or non-mapping file the way a torn
write raises ``JSONDecodeError`` — it returns ``None`` / a bare scalar / a list /
a string, all of which parse *successfully* and then crash the reader's
``.get(...)`` / ``["key"]`` with an uncaught ``AttributeError``/``TypeError``.
The two operator-facing scripts that load config YAML
(``scripts/run_experiment_plan.py`` via ``load_yaml`` and
``scripts/ingest_paper_evidence.py`` via ``load_config``) both had this shape:
an empty or hand-edited config produced a traceback instead of a single readable
error.

What this file pins
-------------------
* **Structural guard** — an AST scan (comments/docstrings ignored) over
  ``src/`` + ``scripts/`` ``.py`` that fails CI on any ``yaml.safe_load`` /
  ``yaml.load`` result dict-accessed (``.get``/``.keys``/``.items``/``.values``/
  ``["key"]``) WITHOUT an ``isinstance(..., dict)`` guard in scope. This makes the
  yaml fix the LAST per-reader patch the way the steering feedback asked the json
  fix to be — a future unguarded ``data = yaml.safe_load(f); data.get(...)``
  reader fails HERE instead of waiting for a crash report.
* **Behavior tests** — the two fixed readers actually tolerate empty / non-dict /
  non-dict-sub-object YAML (clean skip or exit-with-message, never an
  AttributeError). Mutation-proven: on the pre-fix code these raise.

Scope note: the guard watches ``yaml.safe_load`` results accessed DIRECTLY
(chained or via a named var in the same scope). It does NOT trace a value
returned from a wrapper (``config = load_yaml(...)``) — that wrapper-return shape
is closed by making the wrappers themselves guarantee a dict
(``run_experiment_plan.load_yaml`` exits on non-dict; ``inspect_run_config``
guards before access) and is pinned by the behavior tests below. Same deliberate
false-negative bias as the json guard (a guard anywhere in the scope suppresses
the flag) — it exists to catch *regressions* (a brand-new reader with no guard),
not to prove data-flow soundness.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Same non-mutating dict-*reader* access methods as the json guard. None of
# list/str/int/float/bool/None defines them, so each crashes with the SAME
# uncaught AttributeError as ``.get`` on a valid-YAML-but-non-mapping value.
_DICT_ACCESS_ATTRS = frozenset({"get", "keys", "items", "values"})
_SCOPE_STOPPERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _is_yaml_load_call(node: ast.AST) -> bool:
    """True iff *node* is a ``yaml.safe_load(...)`` / ``yaml.load(...)`` call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"safe_load", "load"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "yaml"
    )


def _isinstance_dict_names(nodes: list[ast.AST]) -> set[str]:
    """Bare names guarded by an ``isinstance(<name>, dict)`` under *nodes*.

    Biased toward ``guarded``: any name that is the first arg of an
    ``isinstance(..., dict)`` counts as guarded for the whole scope.
    """
    guarded: set[str] = set()
    for n in nodes:
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "isinstance"
            and len(n.args) >= 2
            and isinstance(n.args[0], ast.Name)
            and isinstance(n.args[1], ast.Name)
            and n.args[1].id == "dict"
        ):
            guarded.add(n.args[0].id)
    return guarded


def _walk_scope(node: ast.AST) -> list[ast.AST]:
    """All nodes in *node*'s scope, NOT descending into nested function/lambda
    bodies (those are separate scopes analyzed independently)."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = [node]
    while stack:
        cur = stack.pop()
        out.append(cur)
        for child in ast.iter_child_nodes(cur):
            if isinstance(child, _SCOPE_STOPPERS):
                continue
            stack.append(child)
    return out


def _is_string_subscript(node: ast.AST) -> bool:
    """``x["key"]`` / ``x['key']`` (string-key subscript = dict access)."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    )


def _yaml_load_assigned_target(value: ast.expr) -> bool:
    """True iff *value* is a direct ``yaml.safe_load(...)`` / ``yaml.load(...)``
    call with args (the whole-file yaml reader shape)."""
    return bool(_is_yaml_load_call(value) and value.args)


def _find_defects(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(line, kind)`` defects in *tree* (a parsed module)."""
    defects: list[tuple[int, str]] = []

    scopes: list[ast.AST] = [tree]
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(n)

    for scope in scopes:
        nodes = _walk_scope(scope)
        guarded = _isinstance_dict_names(nodes)

        yaml_targets: set[str] = set()
        for n in nodes:
            if isinstance(n, ast.Assign) and _yaml_load_assigned_target(n.value):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        yaml_targets.add(t.id)

        for n in nodes:
            # P1: chained yaml.safe_load(...).<dict-method>(...) — no room for guard.
            if (
                isinstance(n, ast.Attribute)
                and n.attr in _DICT_ACCESS_ATTRS
                and _is_yaml_load_call(n.value)
                and n.value.args
            ):
                defects.append(
                    (n.lineno, f"chained yaml.safe_load(...).{n.attr}(...)")
                )
                continue
            # P2a: <yaml_target>.<dict-method>(...)
            if (
                isinstance(n, ast.Attribute)
                and n.attr in _DICT_ACCESS_ATTRS
                and isinstance(n.value, ast.Name)
                and n.value.id in yaml_targets
                and n.value.id not in guarded
            ):
                defects.append(
                    (n.lineno,
                     f"unguarded {n.value.id}.{n.attr}(...) after yaml.safe_load")
                )
                continue
            # P2b: <yaml_target>["key"]
            if (
                _is_string_subscript(n)
                and isinstance(n.value, ast.Name)
                and n.value.id in yaml_targets
                and n.value.id not in guarded
            ):
                defects.append(
                    (n.lineno, f"unguarded {n.value.id}[\"key\"] after yaml.safe_load")
                )

    return defects


def _py_files() -> list[Path]:
    return sorted(
        p for root in (REPO_ROOT / "src", REPO_ROOT / "scripts")
        for p in root.rglob("*.py")
    )


def test_no_unguarded_yaml_load_dict_access_in_py() -> None:
    """Every ``yaml.safe_load``/``yaml.load`` result that is dict-accessed
    (``.get``/``.keys``/``.items``/``.values``/``["key"]``) in ``src/`` +
    ``scripts/`` ``.py`` must be ``isinstance(..., dict)``-guarded."""
    offenders: list[str] = []
    parse_errors: list[str] = []
    for src_file in _py_files():
        text = src_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(src_file))
        except SyntaxError as e:
            parse_errors.append(f"{src_file}:{e.lineno}: {e.msg}")
            continue
        for line, kind in _find_defects(tree):
            offenders.append(f"  {src_file}:{line}: {kind}")

    assert not parse_errors, (
        "A scanned .py file failed to parse — the guard cannot verify it:\n"
        + "\n".join(parse_errors)
    )
    assert not offenders, (
        "Every yaml.safe_load/yaml.load result that is dict-accessed (.get/.keys/"
        ".items/.values/[\"key\"]) must be guarded by isinstance(..., dict) — an "
        "unguarded access crashes with AttributeError on a valid-YAML-but-non-"
        "mapping file (empty→None / bare list/scalar/string), aborting the whole "
        "reader. Add `if not isinstance(data, dict): ...` after the load — see "
        "scripts/run_experiment_plan.py::load_yaml / ingest_paper_evidence.py::"
        "inspect_run_config for the established idiom. Offending sites:\n"
        + "\n".join(offenders)
    )


# --- mutation-proof: the checker itself must catch each defect shape ---

@pytest.mark.parametrize(
    "snippet",
    [
        # P1: chained, no room for a guard.
        "def r(f):\n    return yaml.safe_load(f).get('a')\n",
        "def r(f):\n    return list(yaml.load(f).keys())\n",
        # P2a: named var, .get, no isinstance.
        "def r(f):\n    data = yaml.safe_load(f)\n    return data.get('a')\n",
        # P2a: non-.get dict-reader method crashes identically.
        "def r(f):\n    data = yaml.safe_load(f)\n    return dict(data.items())\n",
        # P2b: named var, string-key subscript, no isinstance.
        "def r(p):\n    data = yaml.safe_load(p.read_text())\n    return data['best']\n",
    ],
)
def test_checker_flags_unguarded_shapes(snippet: str) -> None:
    tree = ast.parse(snippet)
    assert _find_defects(tree) != [], (
        f"guard failed to flag a known defect shape:\n{snippet}"
    )


@pytest.mark.parametrize(
    "snippet",
    [
        # Guarded: isinstance then .get.
        "def r(f):\n    data = yaml.safe_load(f)\n"
        "    return data.get('a') if isinstance(data, dict) else None\n",
        # Guarded inline in the condition.
        "def r(f):\n    data = yaml.safe_load(f)\n"
        "    if isinstance(data, dict) and data.get('a') == 1:\n        pass\n",
        # Guarded, non-.get dict method — must NOT false-positive.
        "def r(f):\n    data = yaml.safe_load(f)\n"
        "    return list(data.keys()) if isinstance(data, dict) else []\n",
        # Not dict-accessed at all — bare return of the load result (the
        # wrapper-return shape; closure of THAT shape is the behavior tests' job).
        "def r(f):\n    return yaml.safe_load(f)\n",
        # Not a yaml load at all — different module name, must not match.
        "def r(f):\n    data = toml.load(f)\n    return data.get('a')\n",
    ],
)
def test_checker_passes_guarded_shapes(snippet: str) -> None:
    tree = ast.parse(snippet)
    assert _find_defects(tree) == [], (
        f"guard false-positive on a guarded/safe shape:\n{snippet}"
    )


# --- behavior: the two fixed readers tolerate non-dict YAML ---

def _load_script_module(name: str, relpath: str):
    """Load a scripts/ file as an isolated module (the scripts are not a package).

    ``ingest_paper_evidence`` imports its sibling ``git_utils`` at module load,
    which needs ``scripts/`` on ``sys.path`` — ensure that before loading.
    """
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.append(str(SCRIPTS_DIR))
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_experiment_plan():
    return _load_script_module("rep_yaml_guard", "scripts/run_experiment_plan.py")


@pytest.fixture(scope="module")
def ingest_paper_evidence():
    return _load_script_module("ipe_yaml_guard", "scripts/ingest_paper_evidence.py")


class TestRunExperimentPlanLoadYaml:
    """``load_yaml`` must guarantee a dict (exit loud on non-dict), not return
    None/a bare value that crashes the caller's ``.get``."""

    def test_returns_dict_for_valid_mapping(self, run_experiment_plan, tmp_path):
        p = tmp_path / "plan.yaml"
        p.write_text("experiment_plan:\n  type: paper-memory\n", encoding="utf-8")
        data = run_experiment_plan.load_yaml(p)
        assert isinstance(data, dict)
        assert data["experiment_plan"]["type"] == "paper-memory"

    def test_exits_on_empty_file(self, run_experiment_plan, tmp_path, capsys):
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")  # yaml.safe_load → None
        with pytest.raises(SystemExit) as ei:
            run_experiment_plan.load_yaml(p)
        assert ei.value.code == 1
        assert "not a top-level mapping" in capsys.readouterr().out

    def test_exits_on_bare_scalar(self, run_experiment_plan, tmp_path, capsys):
        p = tmp_path / "scalar.yaml"
        p.write_text("just-a-string", encoding="utf-8")  # → str
        with pytest.raises(SystemExit) as ei:
            run_experiment_plan.load_yaml(p)
        assert ei.value.code == 1
        assert "not a top-level mapping" in capsys.readouterr().out

    def test_exits_on_top_level_list(self, run_experiment_plan, tmp_path, capsys):
        p = tmp_path / "list.yaml"
        p.write_text("- 1\n- 2\n", encoding="utf-8")  # → list
        with pytest.raises(SystemExit) as ei:
            run_experiment_plan.load_yaml(p)
        assert ei.value.code == 1
        assert "not a top-level mapping" in capsys.readouterr().out


class TestRunExperimentPlanValidateAndPatch:
    """``validate_and_patch_config`` must normalize a non-dict ``experiment:``
    value to a dict so ``exp.get(...)`` can't crash."""

    def test_normalizes_non_dict_experiment_value(
        self, run_experiment_plan, tmp_path
    ):
        cfg = tmp_path / "cfg.yaml"
        # `experiment:` is a scalar — pre-fix, `config['experiment']` then
        # `exp.get(...)` raised AttributeError.
        cfg.write_text("experiment: not-a-dict\n", encoding="utf-8")
        run_experiment_plan.validate_and_patch_config(cfg, "exp-123")
        reloaded = run_experiment_plan.load_yaml(cfg)
        assert isinstance(reloaded["experiment"], dict)
        assert reloaded["experiment"]["paper_experiment"] is True
        assert reloaded["experiment"]["paper_experiment_id"] == "exp-123"

    def test_patches_valid_dict_experiment(self, run_experiment_plan, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "experiment:\n  paper_experiment: false\n", encoding="utf-8"
        )
        run_experiment_plan.validate_and_patch_config(cfg, "exp-456")
        reloaded = run_experiment_plan.load_yaml(cfg)
        assert reloaded["experiment"]["paper_experiment"] is True
        assert reloaded["experiment"]["paper_experiment_id"] == "exp-456"


class TestRunExperimentPlanMainPlanGuard:
    """``main`` must reject a non-dict ``experiment_plan`` value cleanly instead
    of crashing ``plan.get(...)`` on a list/scalar."""

    def test_main_exits_on_non_dict_experiment_plan_value(
        self, run_experiment_plan, tmp_path, monkeypatch, capsys
    ):
        plan = tmp_path / "plan.yaml"
        # Top-level is a dict (passes load_yaml), but `experiment_plan:` is a
        # non-empty list — pre-fix `if not plan:` was False (truthy list) so the
        # code fell through to `plan.get('experiment_id')` → AttributeError.
        plan.write_text("experiment_plan:\n  - 1\n  - 2\n", encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["run_experiment_plan.py", "--config", str(plan)]
        )
        with pytest.raises(SystemExit) as ei:
            run_experiment_plan.main()
        assert ei.value.code == 1
        assert "experiment_plan" in capsys.readouterr().out

    def test_main_exits_on_empty_experiment_plan(
        self, run_experiment_plan, tmp_path, monkeypatch, capsys
    ):
        plan = tmp_path / "plan.yaml"
        plan.write_text("experiment_plan: {}\n", encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["run_experiment_plan.py", "--config", str(plan)]
        )
        with pytest.raises(SystemExit) as ei:
            run_experiment_plan.main()
        assert ei.value.code == 1
        assert "experiment_plan" in capsys.readouterr().out


class TestIngestPaperEvidenceInspectRunConfig:
    """``inspect_run_config`` must skip (return None) on a non-dict config or a
    non-dict ``experiment:`` value, never crash ``exp_meta.get(...)``."""

    def test_none_on_empty_file(self, ingest_paper_evidence, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("", encoding="utf-8")  # → None
        assert ingest_paper_evidence.inspect_run_config(p) is None

    def test_none_on_top_level_scalar(self, ingest_paper_evidence, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("just-a-string", encoding="utf-8")  # → str
        assert ingest_paper_evidence.inspect_run_config(p) is None

    def test_none_on_non_dict_experiment_value(self, ingest_paper_evidence, tmp_path):
        p = tmp_path / "config.yaml"
        # `experiment:` is a scalar — pre-fix, `config['experiment']` then
        # `exp_meta.get(...)` raised AttributeError.
        p.write_text("experiment: not-a-dict\n", encoding="utf-8")
        assert ingest_paper_evidence.inspect_run_config(p) is None

    def test_none_when_not_a_paper_experiment(self, ingest_paper_evidence, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(
            "experiment:\n  paper_experiment: false\n", encoding="utf-8"
        )
        assert ingest_paper_evidence.inspect_run_config(p) is None

    def test_returns_tuple_for_valid_paper_run(self, ingest_paper_evidence, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(
            "experiment:\n  paper_experiment: true\n"
            "  paper_experiment_id: exp-7\n",
            encoding="utf-8",
        )
        assert ingest_paper_evidence.inspect_run_config(p) == (True, "exp-7")
