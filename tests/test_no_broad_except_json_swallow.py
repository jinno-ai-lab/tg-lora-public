"""Static guard: no ``json``/``orjson`` ``load(s)`` behind a BROAD ``except`` in
``scripts/`` or repo-root — the structural closure of the silent-swallow lane.

This is the durable form of the verification the steering feedback asked for after
the FIFTH one-at-a-time promotion of the identical silent-death class
(``scripts/agent_check_status.py`` ``ff0cb31``, following the four sibling readers
``best_run_reader`` / ``run_metrics_reader`` / ``lm_eval_results_reader`` /
``run_footer_reader`` promoted over the prior iterations). The recurring defect:

    try:
        data = json.load(f)          # or json.loads(path.read_text())
    except Exception as e:           # <-- broad: masks the REAL cause
        print(f"[-] Failed to parse: {e}")   # a buried line, or no line at all
        return None                  # <-- silent degrade to a misleading default

A CORRUPT artifact (a torn write, a hand-edited file, a trailing comma) raised
``JSONDecodeError``; the broad ``except Exception`` ate it and the reader
returned ``None`` / ``{}`` / fell through — so corruption became
indistinguishable from ABSENCE, and the downstream consumer acted on a wrong
default. In ``agent_check_status.py`` (the ``make status`` target) that default
drove the operator to "Run the 3-seed paper-memory suite" — burning GPU to
re-run a suite whose summary was merely DAMAGED, never naming the corrupt file.
Each prior iteration promoted ONE reader to a loud, machine-distinguishable
failure (narrow ``except json.JSONDecodeError`` / ``OSError`` → stderr cause
naming the file → ``raise SystemExit(2)``); this guard makes that the LAST
per-reader promotion by failing CI on any *new* broad-except json swallower
instead of waiting for a re-audit.

What it forbids (AST, so comments/docstrings don't false-positive — only real
``try`` statements are walked): a ``json.loads`` / ``json.load`` / ``orjson.loads``
call expression whose lexical enclosing ``try`` has ANY handler that is

* a bare ``except:``, or
* ``except Exception`` / ``except BaseException`` (directly), or
* ``except (... Exception ... BaseException ...)`` (a tuple naming either).

i.e. a broad handler that would mask the ``JSONDecodeError`` / ``OSError`` a
corrupt or unreadable JSON artifact raises.

What it PERMITS (the correct patterns, so the guard is not over-broad):

* **Narrow graceful fallback.** ``frontier_report.py:_read_run_meta`` wraps
  ``json.loads(metadata_path.read_text())`` in
  ``except (json.JSONDecodeError, ValueError, OSError): _read_legacy_files(...)``
  — a deliberate, documented degrade to the source-of-truth legacy files. The
  handler names the *specific* expected failure types, so it does NOT match the
  "broad" detector. Narrow excepts are the sanctioned fallback shape.
* **Scored parse fallback.** ``measure_extraction_fidelity_delta.py`` (the
  headline ``eval_json_extraction`` metric's dual strict→lenient parse) uses
  ``except json.JSONDecodeError: pass`` to fall through to the lenient regex;
  the no-match case is SCORED (``valid=0``), not silently dropped. Narrow →
  permitted.
* **Bare ``json.loads``.** A reader with NO ``try`` at all (the majority —
  ``generate_sweep_dashboard`` / ``export_paper_results`` /
  ``evaluate_paper_gates`` / ``consolidate_paper_results`` /
  ``compare_paper_memory_modes`` / ``analyze_benchmark`` / the JSONL line
  readers) lets ``JSONDecodeError`` propagate → a loud crash + non-zero exit.
  That IS fail-loud (the lane's goal); it is not the polished stderr-cause form
  the operator-facing readers use, but it is not a silent swallow, so it is out
  of this guard's scope.

If a future reader genuinely needs graceful degradation, use a NARROW ``except``
naming the concrete types (``json.JSONDecodeError`` for a malformed file,
``OSError``/``FileNotFoundError`` for an unreadable/missing one) — exactly the
``best_run_reader._fail`` posture — and it will pass this guard. A broad
``except Exception`` around a JSON read is always the anti-pattern this lane
closed.

Out of scope (deliberately, see the report accompanying this guard):
``src/training/train_tg_lora.py`` ~L647 wraps a baseline-metrics ``json.loads``
in a broad ``except Exception`` — but it is (a) inside the production training
loop (not a script/repo-root reader this guard scopes to), (b) an OPTIONAL
best-effort linearity-budget comparison that logs ``logger.warning`` on failure
(not a fully-silent swallow), and (c) verifying any behavior change there needs
a training run this GPU-blocked mirror cannot execute. It is the one known
remainder, deferred to a GPU-available cycle, not churned here.

Attribute-form only (``json.loads(...)`` / ``orjson.loads(...)``): a ``from json
import loads`` bare-name call would evade this guard, but no such import exists
in scope (verified) and ``json.X``/``orjson.X`` is the universal idiom here —
matching the call-expression scoping the sibling ``torch.save`` guard uses.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The JSON deserializer call expressions this lane governs: an attribute access
# ``<jsonmod>.loads`` / ``<jsonmod>.load`` where the receiver is literally named
# ``json`` or ``orjson``. This excludes ``OmegaConf.load`` / ``pickle.load`` /
# ``torch.load`` / ``safetensors`` etc., which are NOT the corrupt-JSON class.
_JSON_MODULES = {"json", "orjson"}


def _handler_is_broad(handler: ast.ExceptHandler) -> bool:
    """True for ``except:``, ``except Exception``, ``except BaseException``, or a
    tuple naming either — the handlers that mask ``JSONDecodeError``/``OSError``."""
    typ = handler.type
    if typ is None:
        return True  # bare ``except:``
    names: list[str] = []

    def _collect(node: ast.expr) -> None:
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                _collect(elt)
        elif isinstance(node, ast.Attribute):
            _collect(node.value)

    _collect(typ)
    return any(n in ("Exception", "BaseException") for n in names)


def _try_has_json_loads(node: ast.Try) -> bool:
    """Does *node*'s body contain a ``json``/``orjson`` ``load(s)`` call?

    Only the ``try`` body is scanned (not the ``except`` handlers/``else``/
    ``finally``): a JSON read that raises is the hazard, and a read in a handler
    is itself guarded by THAT handler's breadth (recursively) when the visitor
    reaches it as its own ``try``.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if (
                child.func.attr in ("loads", "load")
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id in _JSON_MODULES
            ):
                return True
    return False


def _broad_except_json_swallows(root: Path) -> list[tuple[str, int]]:
    """Return ``(rel_path, try_lineno)`` for every broad-except json swallower
    under *root* (a directory of ``.py`` files)."""
    findings: list[tuple[str, int]] = []
    for src_file in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(src_file.read_text(encoding="utf-8"), filename=str(src_file))
        except SyntaxError:
            # An unparseable file is some other test's concern; don't mask it
            # behind a swallow-guard failure.
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Try)
                and any(_handler_is_broad(h) for h in node.handlers)
                and _try_has_json_loads(node)
            ):
                findings.append((str(src_file.relative_to(REPO_ROOT)), node.lineno))
    return findings


def test_no_broad_except_json_swallow_in_scripts() -> None:
    """No ``scripts/`` file may read ``json``/``orjson`` behind a broad ``except``.

    A broad handler (``except:`` / ``except Exception`` / ``except BaseException``)
    around a JSON deserializer masks the ``JSONDecodeError``/``OSError`` a
    corrupt or unreadable artifact raises and silently degrades to a default —
    the exact silent-death class ``ff0cb31`` (+ the four sibling readers)
    promoted out of. This guard holds that closure: a NEW broad-except json
    swallower turns CI red instead of waiting for a re-audit. Use a NARROW
    ``except`` (``json.JSONDecodeError`` / ``OSError``) for any legitimate
    graceful fallback — see the module docstring's permitted-pattern list.
    """
    offenders = _broad_except_json_swallows(REPO_ROOT / "scripts")
    assert not offenders, (
        "broad-except json swallow (the silent-death class ff0cb31 closed) "
        "re-introduced in scripts/:\n"
        + "\n".join(f"  {p}:{ln}" for p, ln in offenders)
        + "\nUse a NARROW except (json.JSONDecodeError / OSError) for a "
        "legitimate fallback, or promote to a fail-loud _fail()+SystemExit like "
        "scripts/best_run_reader.py."
    )


def test_no_broad_except_json_swallow_at_repo_root() -> None:
    """Same invariant as the scripts/ guard, for the repo-root ``.py`` readers
    (the steering feedback's literal 'repo-root/script reader' scope)."""
    offenders = [
        (str(p.relative_to(REPO_ROOT)), ln)
        for p in sorted(REPO_ROOT.glob("*.py"))
        for (rp, ln) in _broad_except_json_swallows(p)
        if p.is_file()
    ]
    assert not offenders, (
        "broad-except json swallow re-introduced at repo root:\n"
        + "\n".join(f"  {p}:{ln}" for p, ln in offenders)
    )
