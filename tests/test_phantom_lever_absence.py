"""Static guard: the recursive-refinement config levers stay OUT of the public mirror.

Why this file exists
--------------------
AI-Hub feedback (this iteration AND a prior one — recorded in PURPOSE.md) asks the
operator to "evaluate flipping ``recursive.enabled`` to default-off (Selection
Rule #5)", i.e. to act on a ``recursive`` / ``max_passes`` "recursive refinement"
feature. That feature does NOT EXIST in this public mirror — it is
private-upstream / AI-Hub-infra vocabulary. A grep of ``src/`` and ``configs/``
for ``recursive`` / ``max_passes`` returns ZERO code hits, and
``scripts/section4_operator_decision.py`` (lines 124-125) documents exactly these
levers as "absent from this public mirror (see TASK-0167..0177)".

Complying with the feedback literally would mean FABRICATING a ``recursive``
config key (and a default flip) to match a phantom reference — a public/private-
boundary violation and a scientific-integrity break (inventing code to satisfy an
infra-side ask instead of advancing real product behavior). The correct response
is to refuse to fabricate, and to make the refusal *structural* rather than a
per-iteration re-discovery: every prior iteration had to re-grep and re-conclude
"phantom" (PURPOSE.md records this), which is exactly the silent drift a static
guard prevents.

This file turns that prose into a durable, mutation-verified contract: the
recursive-refinement levers (``recursive``, ``max_passes``) must not appear as
config keys in ``configs/`` (YAML) or as config-schema fields / config accesses
in ``src/`` Python. A future half-port — or a future agent that complies with the
phantom feedback by inventing the lever — fails CI HERE instead of silently
landing the boundary violation. Same shape as ``test_no_bare_torch_save_in_src``
and the ``test_emitted_json_integrity`` inventory: pin a load-bearing absence so
it cannot regress silently.

Scope is the config-DEFINITION surfaces (``configs/`` YAML keys + ``src/`` Python
field defs / accesses) — the only place a "flip the default" lever can live. It
is NOT a ban on the English word "recursive" in prose: the AST scan ignores
comments and docstrings, and ``TestGuardIsMutationTight`` proves a docstring that
merely *mentions* the lever does NOT trip the guard while a real field / access
DOES (non-vacuity).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The recursive-refinement feature's config keys — the exact levers AI-Hub
# feedback names for a "flip the default" action that cannot land here because the
# feature is absent from this public mirror (documented in
# ``scripts/section4_operator_decision.py`` and verified grep-empty in src/ +
# configs/). Kept to the two levers the feedback ties to the default-flip decision
# (``recursive`` / ``max_passes``); the sibling ``single-pass`` / ``τ=+0.522``
# levers are also absent but are not config-key-shaped, so they stay prose-pinned.
PHANTOM_CONFIG_LEVERS: tuple[str, ...] = ("recursive", "max_passes")

_YAML_KEY_RE = re.compile(
    r"(?m)^[ \t]*(?P<key>"
    + "|".join(re.escape(k) for k in PHANTOM_CONFIG_LEVERS)
    + r")[ \t]*:"
)


def _iter_yaml(repo_root: Path):
    """Yield every config YAML under ``<repo_root>/configs/`` (no subdirs exist)."""
    if not (repo_root / "configs").is_dir():
        return
    yield from repo_root.glob("configs/*.yaml")
    yield from repo_root.glob("configs/*.yml")


def _yaml_phantom_keys(repo_root: Path) -> list[tuple[str, int, str]]:
    """Phantom-lever YAML key defs under ``configs/`` → ``(relpath, lineno, key)``.

    ``recursive:`` / ``max_passes:`` (incl. a nested-indented section key). Whole-
    key only: ``recursive_search:`` does NOT match (the regex requires the colon
    immediately after the lever name, modulo whitespace) — proven by the
    substring-non-vacuity test.
    """
    hits: list[tuple[str, int, str]] = []
    for path in _iter_yaml(repo_root):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for m in _YAML_KEY_RE.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            hits.append((str(path.relative_to(repo_root)), lineno, m.group("key")))
    return hits


def _src_phantom_levers(repo_root: Path) -> list[tuple[str, int, str, str]]:
    """Phantom-lever config usage in ``src/`` Python → ``(relpath, lineno, kind, key)``.

    AST-scanned so comments and docstrings are IGNORED: the guard checks CODE (a
    typed field def, an assignment target, an attribute access, or a construction
    kwarg), never prose. ``recursive`` appearing in a docstring therefore does NOT
    trip it (proven by ``test_ignores_prose_docstring_and_comment``), while
    ``cfg.recursive`` / ``recursive: bool = False`` / ``C(recursive=True)`` DO.
    """
    src = repo_root / "src"
    if not src.is_dir():
        return []
    hits: list[tuple[str, int, str, str]] = []
    for path in sorted(src.rglob("*.py")):
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = str(path.relative_to(repo_root))
        for node in ast.walk(tree):
            # Typed config field:  recursive: bool = False   (dataclass/pydantic)
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in PHANTOM_CONFIG_LEVERS
            ):
                hits.append((rel, node.lineno, "annotated-field", node.target.id))
            # Assignment-target config field:  recursive = False
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in PHANTOM_CONFIG_LEVERS:
                        hits.append((rel, node.lineno, "assignment-target", tgt.id))
            # Config access:  cfg.recursive  /  self.max_passes
            elif isinstance(node, ast.Attribute) and node.attr in PHANTOM_CONFIG_LEVERS:
                hits.append((rel, node.lineno, "attribute-access", node.attr))
            # Config construction kwarg:  TrainingConfig(recursive=True)
            elif isinstance(node, ast.keyword) and node.arg in PHANTOM_CONFIG_LEVERS:
                hits.append((rel, node.lineno, "keyword-arg", node.arg))
    return hits


def test_no_phantom_config_lever_in_configs() -> None:
    """``configs/`` must define no ``recursive`` / ``max_passes`` YAML key."""
    hits = _yaml_phantom_keys(REPO_ROOT)
    assert hits == [], (
        "recursive-refinement config levers must stay out of configs/ — the feature "
        "is AI-Hub/private-infra vocabulary absent from this public mirror (see "
        f"scripts/section4_operator_decision.py). Found YAML keys: {hits}"
    )


def test_no_phantom_config_lever_in_src() -> None:
    """``src/`` must hold no ``recursive`` / ``max_passes`` config field or access."""
    hits = _src_phantom_levers(REPO_ROOT)
    assert hits == [], (
        "recursive-refinement config levers must stay out of src/ — the feature is "
        "absent from this public mirror, so fabricating it to satisfy phantom "
        f"feedback is a boundary violation. Found: {hits}"
    )


class TestGuardIsMutationTight:
    """Prove the guard CATCHES a half-port (non-vacuity) and IGNORES prose.

    A guard that passes because it scans nothing is worthless. These pin that a
    fabricated ``recursive`` / ``max_passes`` field, YAML key, attribute access,
    or kwarg trips the guard, while a docstring/comment merely *mentioning* the
    lever does not, and a similarly-spelled key (``recursive_search``) does not.
    """

    def test_catches_recursive_yaml_key(self, tmp_path: Path) -> None:
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "x.yaml").write_text("recursive:\n  enabled: false\n")
        assert _yaml_phantom_keys(tmp_path) == [("configs/x.yaml", 1, "recursive")]

    def test_catches_nested_max_passes_yaml_key(self, tmp_path: Path) -> None:
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "x.yaml").write_text("model:\n  max_passes: 3\n")
        assert _yaml_phantom_keys(tmp_path) == [("configs/x.yaml", 2, "max_passes")]

    def test_yaml_whole_key_only_not_substring(self, tmp_path: Path) -> None:
        # ``recursive_search:`` must NOT match — whole-key discipline.
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "x.yaml").write_text("recursive_search: true\n")
        assert _yaml_phantom_keys(tmp_path) == []

    def test_catches_src_annotated_field(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "m.py").write_text("class C:\n    recursive: bool = False\n")
        hits = _src_phantom_levers(tmp_path)
        assert len(hits) == 1
        assert hits[0][2] == "annotated-field" and hits[0][3] == "recursive"

    def test_catches_src_attribute_access(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "m.py").write_text("def f(cfg):\n    return cfg.recursive\n")
        hits = _src_phantom_levers(tmp_path)
        assert len(hits) == 1
        assert hits[0][2] == "attribute-access" and hits[0][3] == "recursive"

    def test_catches_src_kwarg(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "m.py").write_text("C(max_passes=3)\n")
        hits = _src_phantom_levers(tmp_path)
        assert len(hits) == 1
        assert hits[0][2] == "keyword-arg" and hits[0][3] == "max_passes"

    def test_ignores_prose_docstring_and_comment(self, tmp_path: Path) -> None:
        # Non-vacuity: prose mentioning the levers must NOT trip the guard. If this
        # fails, the guard is scanning text (not code) and is useless as a
        # boundary check.
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "m.py").write_text(
            '"""The recursive refinement lever (recursive.enabled / max_passes)\n'
            'is absent from this mirror — do not fabricate it."""\n'
            "# note: recursive / max_passes are phantom AI-Hub vocabulary\n"
            "value = 1\n"
        )
        assert _src_phantom_levers(tmp_path) == []
