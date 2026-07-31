"""Structural isolation guard: the §4 Cat-C significance verdict is computed
from valid_loss ONLY — never from the eval JSON-extraction metric.

AI-Hub make-run feedback (this iter, bullet 4) asked to *quantify* whether the
JSON-extraction metric's "false-invalid" (a valid-JSON model output the greedy
parser scored as invalid) biased prior Cat-C significance scores:

    "if it systematically favored clean-JSON models it may have moved a measured
    split and warrants a targeted re-measure; otherwise park eval-pipeline work."

The answer is structural, not statistical — the two are on DISJOINT data paths:

  * **Cat-C significance** = :func:`bootstrap_difference_ci` /
    :func:`surrogate_valid_loss_ci` → a percentile bootstrap on
    ``mean(surrogate_valid_loss) − mean(candidate_valid_loss)``. Every input is
    a valid_loss (cross-entropy) sequence deposited by the GPU run; the
    ``point_improvement`` / ``is_material`` / ``significance_verdict`` fields
    the §4 deposit carries are all derived from those valid_loss statistics.
  * **false-invalid** = :func:`evaluate_json_extraction_run`
    (``src.eval.jsonex_generation``), the headline *quality* metric. In the
    training loop its return (``gold_scores``) is recorded only as side-channel
    ``gold_*`` extra-fields on ``metrics.record_step`` and appended to
    ``json_eval_log.jsonl`` (``src/training/train_tg_lora.py``) — it never feeds
    ``best_valid_loss``, the freeze controller, rollback, or the significance CI.

So the false-invalid cannot have biased the Cat-C significance verdict: there
is no edge from the JSON metric into the valid_loss significance path, hence
no "clean-JSON-favoured" split to re-measure. Eval-pipeline work is parked.

This guard pins that absence **structurally** (AST, so a comment or docstring
cannot mask a future coupling) so the bias failure mode can never be
*introduced* — the same load-bearing-absence shape as
``test_phantom_lever_absence`` / ``test_no_bare_torch_save_in_src``: if someone
wires the JSON metric into a significance-path module, this goes RED with a
file/line/kind diagnostic. Mutation-proven end-to-end (see commit message).
"""

import ast
import dataclasses
import inspect
from pathlib import Path

from src.tg_lora import freeze_surrogate_ci as ci_mod
from src.tg_lora import freeze_surrogate_gate as gate_mod
from src.tg_lora.freeze_surrogate_ci import (
    SurrogateValidLossCI,
    bootstrap_difference_ci,
    surrogate_valid_loss_ci,
)

# The modules whose output determines the §4 Cat-C significance verdict: the
# bootstrap CI (ci) and the structural surrogate-exceedance gate it promotes.
SIGNIFICANCE_MODULES = (ci_mod, gate_mod)

# The eval-package coupling that would let a JSON-metric false-invalid reach
# the valid_loss significance verdict. Unambiguous: ``src.eval`` is entirely the
# JSON/quality-metric package (eval_json_extraction / jsonex_generation /
# json_generation), and ``evaluate_json_extraction_run`` is its entrypoint —
# there is no non-JSON reason for a significance module to import from there.
_FORBIDDEN_IMPORT_STEMS = (
    "src.eval",
    "eval.eval_json_extraction",
    "eval.jsonex_generation",
    "eval.json_generation",
)
_FORBIDDEN_REFERENCES = ("evaluate_json_extraction_run",)

# Tokens that would betray a JSON/quality-metric field or parameter sneaking
# into the valid_loss-only significance contract.
_JSON_METRIC_TOKENS = ("json", "gold", "extract", "accuracy", "f1")


def _looks_like_json_metric(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in _JSON_METRIC_TOKENS)


def _module_path(module) -> Path:
    return Path(inspect.getfile(module))


def _parse(module):
    path = _module_path(module)
    return path, ast.parse(path.read_text(), filename=str(path))


def test_significance_modules_do_not_import_the_eval_package():
    """No significance-path module may import from src.eval (the JSON metric)."""
    offenders: list[str] = []
    for module in SIGNIFICANCE_MODULES:
        _path, tree = _parse(module)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if any(
                    node.module == stem or node.module.startswith(stem + ".")
                    for stem in _FORBIDDEN_IMPORT_STEMS
                ):
                    offenders.append(
                        f"{module.__name__}:{node.lineno} imports {node.module!r}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == stem or alias.name.startswith(stem + ".")
                        for stem in _FORBIDDEN_IMPORT_STEMS
                    ):
                        offenders.append(
                            f"{module.__name__}:{node.lineno} imports {alias.name!r}"
                        )
    assert not offenders, (
        "§4 Cat-C significance must stay isolated from the eval JSON-extraction "
        "metric (the false-invalid source). Coupling found — a JSON-metric "
        "false-invalid could now bias the valid_loss significance verdict:\n  "
        + "\n  ".join(offenders)
    )


def test_significance_modules_never_reference_the_json_extraction_entrypoint():
    """No Name/Attribute/def reference to evaluate_json_extraction_run."""
    offenders: list[str] = []
    for module in SIGNIFICANCE_MODULES:
        _path, tree = _parse(module)
        for node in ast.walk(tree):
            ref = None
            if isinstance(node, ast.Name):
                ref = node.id
            elif isinstance(node, ast.Attribute):
                ref = node.attr
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ref = node.name
            if ref in _FORBIDDEN_REFERENCES:
                offenders.append(
                    f"{module.__name__}:{node.lineno} references {ref!r}"
                )
    assert not offenders, (
        "§4 Cat-C significance must not call or alias the JSON-extraction "
        "entrypoint. References found:\n  " + "\n  ".join(offenders)
    )


def test_significance_ci_takes_only_valid_loss_inputs():
    """bootstrap_difference_ci / surrogate_valid_loss_ci take valid_loss sequences.

    A positive contract: the headline significance APIs are parameterised on
    loss sequences, never on a JSON/quality score. Asserts both the expected
    valid_loss parameter IS present and no JSON-metric parameter snuck in.
    """
    contracts = (
        (bootstrap_difference_ci, "candidate_losses"),
        (bootstrap_difference_ci, "surrogate_losses"),
        (surrogate_valid_loss_ci, "candidate_valid_losses"),
        (surrogate_valid_loss_ci, "surrogate_valid_losses"),
    )
    for func, param in contracts:
        params = inspect.signature(func).parameters
        assert param in params, (
            f"{func.__qualname__} lost its valid_loss parameter {param!r} — the "
            "significance CI must be driven by valid_loss sequences."
        )
        for name in params:
            assert not _looks_like_json_metric(name), (
                f"{func.__qualname__} gained a JSON-metric parameter {name!r}; "
                "the §4 significance CI is valid_loss-only."
            )


def test_significance_verdict_dataclass_carries_only_valid_loss_statistics():
    """SurrogateValidLossCI fields are valid_loss statistics, not JSON scores."""
    fields = {f.name for f in dataclasses.fields(SurrogateValidLossCI)}
    for name in fields:
        assert not _looks_like_json_metric(name), (
            f"SurrogateValidLossCI gained a JSON-metric field {name!r}; the §4 "
            "significance verdict must be derivable from valid_loss alone."
        )
    # The headline valid_loss statistics that drive the verdict must be present
    # as stored fields (is_material / point_improvement are derived @property
    # off candidate_mean + surrogate_mean + material_margin, not stored fields).
    assert {
        "candidate_mean",
        "surrogate_mean",
        "lower",
        "upper",
        "significance_verdict",
        "material_margin",
    } <= fields, (
        "SurrogateValidLossCI lost a load-bearing valid_loss statistic; the §4 "
        f"significance contract changed: {sorted(fields)}"
    )
