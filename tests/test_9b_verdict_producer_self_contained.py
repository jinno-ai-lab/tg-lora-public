"""Static guard: the 9B §4 verdict producer stays self-contained — no private
``src.data`` dependency — so the verdict run is independently bootable on the
public mirror.

Why this file exists
--------------------
A recurring false narrative — "the 9B §4 run is DATA-blocked (private ``src.data``
absent) AND GPU-blocked (12 GB OOM)" — drove 30+ proxy/peripheral iterations and
three consecutive ``no-op-with-blocker`` TASK docs (TASK-0236 / 0237 / 0238, all
2026-08-06). Both prongs are FALSE for the verdict producer:

* **Not DATA-blocked.** ``scripts/run_freeze_validloss_ci_9b.py`` is explicitly
  src.data-free — its public SFT adapter trains on
  ``databricks/databricks-dolly-15k`` and the module docstring states "No private
  ``src.data`` is used". The launcher ``scripts/launch_freeze_ci_9b_full.py`` is
  stdlib-only. This guard AST-verifies the contract: a real
  ``import src.data`` / ``from src.data import …`` in either entry point fails CI.
* **Not GPU-blocked at seq1024.** The suffix-only seq1024 QLoRA stack is
  engineered to fit a sustained run on 12 GB (``PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True`` + the load-once / reset arm-separation primitive
  that avoids a second ~5.5 GB 9B model) — which is why the committed
  ``tests/fixtures/freeze_validloss_ci_9b_full*.json`` deposits are real RTX 3060
  seq1024 runs and ``tests/test_run_freeze_validloss_ci_9b.py`` documents the GPU
  run as an opt-in smoke, not an impossibility.
* **Empirically re-confirmed 2026-08-06.** A 20-step smoke
  (``--total-steps 20 --n-candidate 1 --n-surrogate 1 --seq-len 1024``) booted
  end-to-end on the idle 12 GB RTX 3060 in ~70 s — loaded ``Qwen/Qwen3.5-9B``
  (427 shards, 8.997 B params), tokenized public Dolly, trained candidate +
  surrogate, and wrote a deposit with ``citable_as_target_scale=True`` and no OOM.

Each prior iteration re-derived "blocked" in prose (TASK-0236/0237/0238 each
re-grep and re-conclude). This guard turns the src.data-free refutation into a
durable, mutation-verified contract so the DATA-blocked prong of the narrative
cannot recur — the same shape as ``test_phantom_lever_absence.py``, but for the
inverse claim (a path that IS present and bootable, not a lever that is absent).

Scope
-----
The two entry points ``make freeze-validloss-ci-9b-full*`` actually boots:
``scripts/run_freeze_validloss_ci_9b.py`` (producer) and
``scripts/launch_freeze_ci_9b_full.py`` (self-retrying background launcher).
Sibling scripts that DO import ``src.data`` (``prepare_data.py``,
``collect_true_gradients.py``, ``offline_*.py``, …) are the private-pipeline-
dependent tooling stripped from this mirror — they are NOT on the verdict path.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The entry points `make freeze-validloss-ci-9b-full*` boots — the producer and
# its self-retrying background launcher. These are what must stay src.data-free
# for the verdict run to be independently bootable on the public mirror.
VERDICT_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/run_freeze_validloss_ci_9b.py",
    "scripts/launch_freeze_ci_9b_full.py",
)


def _src_data_imports(rel: str) -> list[tuple[int, str, str]]:
    """Real ``src.data`` import statements in one file → ``(lineno, kind, name)``.

    AST-scanned so docstrings and comments are IGNORED — the producer's docstring
    *says* "No private ``src.data`` is used"; that prose does not trip the guard.
    Only a real ``import src.data[.…]`` (Import) or ``from src.data[.…] import …``
    (ImportFrom) does. ``from src.model import …`` / ``from src.tg_lora import …``
    / ``from src.training import …`` do NOT match — the verdict path legitimately
    uses the public in-repo ``src`` modules; only the stripped private ``src.data``
    pipeline is forbidden.
    """
    path = REPO_ROOT / rel
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "src.data" or alias.name.startswith("src.data."):
                    hits.append((node.lineno, "import", alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "src.data" or module.startswith("src.data."):
                hits.append((node.lineno, "from", module))
    return hits


def test_9b_verdict_entry_points_have_no_src_data_dependency() -> None:
    """The 9B §4 verdict entry points must not import private ``src.data``.

    This is the structural refutation of the "9B run is DATA-blocked" narrative:
    while the entry points stay src.data-free, the verdict run is independently
    bootable on the public mirror (re-confirmed by the 2026-08-06 RTX 3060 smoke).
    A regression here would re-open the false blocker that drove the proxy-iteration
    detour.
    """
    all_hits: list[tuple[str, int, str, str]] = []
    for rel in VERDICT_ENTRY_POINTS:
        for lineno, kind, name in _src_data_imports(rel):
            all_hits.append((rel, lineno, kind, name))
    assert all_hits == [], (
        "9B §4 verdict entry points must stay src.data-free so the run is "
        "independently bootable on the public mirror (not DATA-blocked). Found "
        f"src.data imports: {all_hits}"
    )
