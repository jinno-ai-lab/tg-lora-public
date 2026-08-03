"""Static guard: every ``json/orjson.load(s)`` → ``.get`` / ``["key"]`` access is
``isinstance(..., dict)``-guarded — close the non-dict-after-json.loads/load crash class.

This is the structural closure the steering feedback asked for after the THIRD
one-at-a-time patch of the identical crash class (compare_runs.load_run 374468d,
analyze_json_experiment/analyze_trajectory 6b1da5a, the §4 verdict ledger readers
38838a0, the sweep-summary readers 51d1975). The recurring defect: a line/file
that is *valid JSON but not an object* (a bare array/scalar/string/number/null)
parses fine from ``json.loads``, yet the reader immediately does ``obj.get(...)``
or ``obj["key"]`` on the result expecting a dict — so the non-dict value raises
an uncaught ``AttributeError: 'list' object has no attribute 'get'`` that aborts
the whole summary/query/resume. Each prior iteration hardened one reader; this
guard makes that the LAST per-reader patch by failing CI on any *new* unguarded
reader instead of waiting for a crash report.

What it catches (AST, so comments/docstrings don't false-positive — only real
expressions) — every shape of the non-dict-after-json.loads crash class, not
just the dominant ``.get`` shape:

* **P1 (chained)** — ``json.loads(x).get(...)`` / ``.keys()`` / ``.items()`` /
  ``.values()`` / ``["k"]``: the loads result is dict-accessed inline with no
  place to put a guard. Always a defect; refactor to
  ``obj = loads(x); if isinstance(obj, dict) and obj.get(...)``.
* **P2 (named var)** — ``obj = loads(...)`` (or ``recs = [loads(l) for l in f]``
  with NO ``isinstance`` filter in the comprehension) whose value is later
  ``.get()`` / ``.keys()`` / ``.items()`` / ``.values()`` / ``["str"]`` /
  ``[i].get()``-accessed without an ``isinstance(obj, dict)`` check anywhere in
  the enclosing scope. The dict-*reader* methods (get/keys/items/values) are all
  watched: none of list/str/int/bool/None defines them, so each crashes with the
  same uncaught ``AttributeError`` on a valid-JSON-but-non-object value.

Scopes are walked per-function (nested function bodies are analyzed
independently, not folded into the parent). The ``isinstance`` guard is matched
*per scope* and biased toward NOT flagging (a name is treated as guarded for the
whole scope once any ``isinstance(name, dict)`` on it exists there) — the guard
exists to catch *regressions* (a brand-new reader with no guard at all), not to
prove data-flow soundness, so it deliberately trades a few false negatives for
zero false positives.

Scope is ``src/`` + ``scripts/`` ``.py`` files (the crash class spans both), plus
a best-effort line-scan of ``scripts/*.sh`` for the same pattern in shell-embedded
``python -c`` / heredoc blocks (the readers fixed alongside this guard lived
there and were missed by the earlier ``.py``-only patches). BOTH per-line JSONL
readers AND whole-file ``data = loads(path.read_text())`` readers are in scope:
the original carve-out assumed a whole-file load is dict-by-construction (a torn
write raises ``JSONDecodeError``), but a *valid-JSON-but-non-object* whole file
(bare array/scalar/string — a hand-edited, externally-written, or wrong-format
file) parses fine and then crashes ``data.get(...)`` / ``data["k"]`` with the same
uncaught ``AttributeError``/``TypeError``. The whole-file readers fixed in the
comprehensive sweep (consolidate / evaluate_paper_gates / frontier_report /
precompute / deterministic_batch_plan / export_paper_results / compare_paper_*
+ the ``*.sh`` summary loaders) proved this is a real, not theoretical, failure
mode, so the exclusion was removed — the guard now enforces "every reader" as
the steering feedback asked.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_JSON_NAMES = {"json", "orjson"}
# The non-mutating dict-*reader* access methods. A loads result is one of the
# JSON value types (dict / list / str / int / float / bool / None); NONE of
# list / str / int / float / bool / None defines ``.keys`` / ``.items`` /
# ``.values``, so calling any of these on a valid-JSON-but-non-object value
# raises the SAME uncaught ``AttributeError`` as ``.get`` and aborts the whole
# reader. The guard watches all four (not just ``.get``) so the structural
# closure covers every shape of the non-dict-after-json.loads crash class, not
# merely the dominant ``.get`` shape — a ``.keys()`` reader is a real pattern
# (two guarded sites in eval_downstream / eval_format) that would otherwise need
# its own per-reader patch, defeating the guard's "last per-reader patch"
# purpose. Mutating methods (pop / setdefault / popitem / update / ...) and
# names that also exist on list (pop / copy / clear) are deliberately OUT OF
# SCOPE: they don't fit the unambiguous non-mutating-reader shape and would risk
# false positives.
_DICT_ACCESS_ATTRS = frozenset({"get", "keys", "items", "values"})
_SCOPE_STOPPERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _is_loads_call(node: ast.AST) -> bool:
    """True iff *node* is a ``json``/``orjson`` ``.loads(...)`` or ``.load(...)``
    call. Both spellings are the SAME non-dict-after-parse crash class:
    ``json.load(f)`` (whole-file / file-object form) parses a valid-JSON-but-
    non-object just as ``json.loads(s)`` does, so a subsequent
    ``.get``/``.keys``/``["k"]`` crashes identically. Matching only ``loads``
    left the ``load`` spelling as a structural blind spot — it escaped every
    prior per-reader pass and the comprehensive sweep."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("load", "loads")
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in _JSON_NAMES
    )


def _isinstance_dict_names(nodes: list[ast.AST]) -> set[str]:
    """Bare names guarded by an ``isinstance(<name>, dict)`` under *nodes*.

    Biased toward ``guarded``: any name that is the first arg of an
    ``isinstance(..., dict)`` (incl. inside an ``and``/``not``) counts as guarded
    for the whole scope, so a real guard anywhere in the function suppresses the
    flag everywhere in it.
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


def _comp_has_dict_filter(comp: ast.AST) -> bool:
    """True iff a comprehension carries an ``isinstance(..., dict)`` if-filter."""
    for gen in getattr(comp, "generators", []):
        for ifx in gen.ifs:
            if _isinstance_dict_guard_single(ifx):
                return True
    return False


def _isinstance_dict_guard_single(node: ast.AST) -> bool:
    return bool(_isinstance_dict_names(_walk_scope(node)))


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
    """``x["key"]`` / ``x['key']`` (string-key subscript = dict access). Integer
    subscripts (``x[0]`` list access) deliberately do NOT match."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    )


def _loads_assigned_target(value: ast.expr) -> bool:
    """True iff *value* yields raw loads output that may contain non-dict
    elements: a direct ``loads(...)`` call (per-line OR whole-file — both can
    parse a valid-JSON non-object), or a comprehension of ``loads(...)`` with NO
    ``isinstance`` dict filter (``[loads(l) for l in f]``)."""
    if _is_loads_call(value) and value.args:
        return True
    if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        if _is_loads_call(value.elt) and not _comp_has_dict_filter(value):
            return True
    return False


def _find_defects(tree: ast.AST, filename: str) -> list[tuple[int, str]]:
    """Return ``(line, kind)`` defects in *tree* (a parsed module)."""
    defects: list[tuple[int, str]] = []

    # Every scope-defining node: the module itself + each function.
    scopes: list[ast.AST] = [tree]
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(n)

    for scope in scopes:
        nodes = _walk_scope(scope)

        # Names guarded by isinstance(..., dict) somewhere in THIS scope.
        guarded = _isinstance_dict_names(nodes)

        # Names whose value is raw per-line loads output (direct call or bare
        # comp), plus loop variables iterating one of those
        # (``for r in recs: r.get(...)``). Collected to a fixpoint so source order
        # is irrelevant — ``_walk_scope`` yields nodes in DFS, not source, order.
        loads_targets: set[str] = set()
        for n in nodes:
            if isinstance(n, ast.Assign) and _loads_assigned_target(n.value):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        loads_targets.add(t.id)
        changed = True
        while changed:
            changed = False
            for n in nodes:
                if (
                    isinstance(n, ast.For)
                    and isinstance(n.iter, ast.Name)
                    and n.iter.id in loads_targets
                    and isinstance(n.target, ast.Name)
                    and n.target.id not in loads_targets
                ):
                    loads_targets.add(n.target.id)
                    changed = True

        for n in nodes:
            # P1: chained loads(...).<dict-method>(...) — no room for a guard, always bad.
            if (
                isinstance(n, ast.Attribute)
                and n.attr in _DICT_ACCESS_ATTRS
                and _is_loads_call(n.value)
                and n.value.args
            ):
                defects.append(
                    (n.lineno, f"chained json.loads(...).{n.attr}(...)")
                )
                continue
            # P2a: <loads_target>.<dict-method>(...)
            if (
                isinstance(n, ast.Attribute)
                and n.attr in _DICT_ACCESS_ATTRS
                and isinstance(n.value, ast.Name)
                and n.value.id in loads_targets
                and n.value.id not in guarded
            ):
                defects.append(
                    (n.lineno,
                     f"unguarded {n.value.id}.{n.attr}(...) after json.loads")
                )
                continue
            # P2b: <loads_target>["key"]
            if (
                _is_string_subscript(n)
                and isinstance(n.value, ast.Name)
                and n.value.id in loads_targets
                and n.value.id not in guarded
            ):
                defects.append(
                    (n.lineno, f"unguarded {n.value.id}[\"key\"] after json.loads")
                )
                continue
            # P2c: <loads_target>[i].<dict-method>(...) (list of loads records)
            if (
                isinstance(n, ast.Attribute)
                and n.attr in _DICT_ACCESS_ATTRS
                and isinstance(n.value, ast.Subscript)
                and isinstance(n.value.value, ast.Name)
                and n.value.value.id in loads_targets
                and n.value.value.id not in guarded
            ):
                defects.append(
                    (n.lineno,
                     f"unguarded {n.value.value.id}[...].{n.attr}(...) after json.loads")
                )

    return defects


def _py_files() -> list[Path]:
    return sorted(
        [*(
            p for root in (REPO_ROOT / "src", REPO_ROOT / "scripts")
            for p in root.rglob("*.py")
        )]
    )


def test_no_unguarded_loads_dict_access_in_py() -> None:
    """Every ``json/orjson.load(s)`` result that is dict-accessed
    (``.get``/``.keys``/``.items``/``.values``/``["key"]``) in ``src/`` +
    ``scripts/`` ``.py`` must be ``isinstance(..., dict)``-guarded. Both the
    ``loads`` (string) and ``load`` (file-object) forms are matched — same
    non-dict-after-parse crash class."""
    offenders: list[str] = []
    parse_errors: list[str] = []
    for src_file in _py_files():
        text = src_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(src_file))
        except SyntaxError as e:
            parse_errors.append(f"{src_file}:{e.lineno}: {e.msg}")
            continue
        for line, kind in _find_defects(tree, str(src_file)):
            offenders.append(f"  {src_file}:{line}: {kind}")

    assert not parse_errors, (
        "A scanned .py file failed to parse — the guard cannot verify it:\n"
        + "\n".join(parse_errors)
    )
    assert not offenders, (
        "Every json/orjson.loads result that is dict-accessed (.get/.keys/.items/"
        ".values/[\"key\"]) must be guarded by isinstance(..., dict) — an unguarded "
        "access crashes with AttributeError on a valid-JSON-but-non-object line "
        "(bare array/scalar/string), aborting the whole reader. Add "
        "`if not isinstance(obj, dict): continue` (or filter in the comprehension) "
        "— see io.load_jsonl / parse_jsonl / compare_runs.load_run for the "
        "established idiom. Offending sites:\n" + "\n".join(offenders)
    )


# --- shell-embedded python: the six readers fixed here lived in scripts/*.sh ---

# ``loads?`` matches BOTH ``json.loads(`` (string form) and ``json.load(``
# (file-object form) — same non-dict crash class (see ``_is_loads_call``): a
# valid-JSON-but-non-object file parses either way, then ``.get``/``[]`` crashes.
_NAMED_LOADS_RE = re.compile(r"(\w+)\s*=\s*(?:json|orjson)\.loads?\(")
# Chained load(s)(...).<dict-reader-method>(...) — get/keys/items/values all
# crash identically on a non-dict (none exist on list/str/int/bool/None).
_CHAINED_LOADS_ACCESS_RE = re.compile(
    r"(?:json|orjson)\.loads?\([^)\n]*\)\s*\.\s*(?:get|keys|items|values)\("
)


def _sh_defects(path: Path) -> list[tuple[int, str]]:
    """Best-effort scan of shell-embedded python for the same crash class.

    Catches the chained form (``json.loads(line).get('type')`` / ``.keys()`` /
    ``.items()`` / ``.values()``) and the named-var form (``obj =
    json.loads(line)`` on one line, ``obj.get(...)`` / ``obj.keys()`` /
    ``obj['x']`` on the next, with no ``isinstance(obj, dict)`` nearby). Shell
    quoting makes a full extraction brittle, so this is a line-scan — but it
    covers the exact shapes the six fixed .sh readers used, generalized from
    ``.get`` to the full non-mutating dict-reader method set so a future embedded
    ``.keys()`` reader can't slip through either.
    """
    defects: list[tuple[int, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        # Commented-out code (leading ``#``) never executes — it can't crash, so
        # the line-scan must not flag a ``load(s)`` reader that lives only in a
        # comment (e.g. an illustrative ``#   d=json.load(open(p))`` recipe).
        if line.lstrip().startswith("#"):
            continue
        if _CHAINED_LOADS_ACCESS_RE.search(line):
            defects.append((i + 1, "chained json.load(s)(...).<dict-method>(...)"))
        m = _NAMED_LOADS_RE.search(line)
        if m:
            var = m.group(1)
            window = "\n".join(lines[i : i + 4])
            guarded = f"isinstance({var}, dict)" in window or "isinstance" in window
            accessed = bool(
                re.search(
                    rf"{re.escape(var)}\s*\.\s*(?:get|keys|items|values)\(", window
                )
                or re.search(rf"{re.escape(var)}\s*\[\s*['\"]", window)
            )
            if accessed and not guarded:
                defects.append(
                    (i + 1, f"unguarded {var}.<dict-method>/[] after json.load(s)")
                )
    return defects


def test_no_unguarded_loads_dict_access_in_sh() -> None:
    """Shell-embedded python in ``scripts/*.sh`` must guard ``load(s)`` → ``.get``."""
    offenders: list[str] = []
    for sh in sorted((REPO_ROOT / "scripts").glob("*.sh")):
        for line, kind in _sh_defects(sh):
            offenders.append(f"  {sh}:{line}: {kind}")
    assert not offenders, (
        "Shell-embedded python must guard json/orjson.loads → .get/[] the same way "
        ".py readers do (a non-dict JSON line crashes the embedded block and the "
        "make target it backs). Offending sites:\n" + "\n".join(offenders)
    )


# --- mutation-proof: the checker itself must catch each defect shape ---

@pytest.mark.parametrize(
    "snippet",
    [
        # P1: chained, no room for a guard.
        "for line in f:\n    if json.loads(line).get('type') == 'step':\n        pass\n",
        "x = orjson.loads(b'[1,2]').get('a')\n",
        # P2a: named var, .get, no isinstance.
        "def r(f):\n    for line in f:\n        obj = json.loads(line)\n"
        "        if obj.get('type') == 'step':\n            pass\n",
        # P2b: named line var, string-key subscript, no isinstance.
        "def r(f):\n    for line in f:\n        obj = json.loads(line)\n        return obj['best']\n",
        # P2c: list of loads records, then [i].get — no filter in the comp.
        "def r(f):\n    recs = [json.loads(l) for l in f]\n"
        "    return recs[0].get('type')\n",
        # loop var over a loads list, then .get.
        "def r(f):\n    recs = [json.loads(l) for l in f]\n"
        "    for r in recs:\n        r.get('type')\n",
        # Whole-file load (named var), .get, no isinstance — now in scope
        # (a valid-JSON-but-non-object file crashes .get just like a per-line).
        "def r(p):\n    data = json.loads(p.read_text())\n    return data.get('x')\n",
        # Whole-file load, chained .get — no room for a guard; now in scope.
        "def r(p):\n    return json.loads(p.read_text()).get('x')\n",
        # Non-.get dict-reader methods on a loads result crash identically
        # (list/str/int/bool/None define no .keys/.items/.values) — the same
        # crash class, so the guard must flag them, not just .get. Named-var form.
        "def r(p):\n    obj = json.loads(p.read_text())\n    return list(obj.keys())\n",
        "def r(p):\n    obj = json.loads(p.read_text())\n    return dict(obj.items())\n",
        "def r(p):\n    obj = json.loads(p.read_text())\n    return sum(1 for _ in obj.values())\n",
        # Non-.get dict method, chained — no room for a guard.
        "def r(p):\n    return list(json.loads(p.read_text()).keys())\n",
        # List of loads records, then [i].items() — no isinstance filter in the comp.
        "def r(f):\n    recs = [json.loads(l) for l in f]\n    return dict(recs[0].items())\n",
        # json.load (file-object form) — SAME crash class as loads: a valid-JSON
        # non-object file parses, then .get crashes. The ``load`` spelling must
        # not escape the guard. Named-var form.
        "def r(f):\n    data = json.load(f)\n    return data.get('x')\n",
        # json.load, chained .get — no room for a guard; must still flag.
        "def r(f):\n    return json.load(f).get('x')\n",
    ],
)
def test_checker_flags_unguarded_shapes(snippet: str) -> None:
    tree = ast.parse(snippet)
    assert _find_defects(tree, "<test>") != [], (
        f"guard failed to flag a known defect shape:\n{snippet}"
    )


@pytest.mark.parametrize(
    "snippet",
    [
        # Guarded: isinstance then .get.
        "def r(f):\n    for line in f:\n        obj = json.loads(line)\n"
        "        if not isinstance(obj, dict):\n            continue\n"
        "        if obj.get('type') == 'step':\n            pass\n",
        # Guarded inline in the condition.
        "def r(f):\n    for line in f:\n"
        "        obj = json.loads(line)\n"
        "        if isinstance(obj, dict) and obj.get('type') == 'step':\n            pass\n",
        # Filtered comprehension — element is filtered, downstream .get is safe.
        "def r(f):\n    recs = [r for r in (json.loads(l) for l in f) if isinstance(r, dict)]\n"
        "    return recs[0].get('type')\n",
        # Not dict-accessed at all — opaque list round-trip (no .get on loads result).
        "def r(f):\n    recs = [json.loads(l) for l in f]\n    return len(recs)\n",
        # Integer subscript is list access, not a dict access — not flagged.
        "def r(f):\n    recs = [json.loads(l) for l in f]\n    return recs[0]\n",
        # Whole-file load guarded by isinstance.
        "def r(p):\n    data = json.loads(p.read_text())\n"
        "    return data.get('x') if isinstance(data, dict) else None\n",
        # Non-.get dict method, guarded by isinstance — must NOT false-positive.
        "def r(p):\n    obj = json.loads(p.read_text())\n"
        "    return list(obj.keys()) if isinstance(obj, dict) else []\n",
    ],
)
def test_checker_passes_guarded_shapes(snippet: str) -> None:
    tree = ast.parse(snippet)
    assert _find_defects(tree, "<test>") == [], (
        f"guard false-positive on a guarded/safe shape:\n{snippet}"
    )
