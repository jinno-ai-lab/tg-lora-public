"""Regression test: the dynfreeze all-frozen skip path must not crash.

When ``DynamicFreezeController.run_cycle`` reports every layer frozen
(``dynfreeze_all_frozen == True``), the cycle skips the pilot / extrapolation /
post-eval and records metrics directly. The skip block
(``train_tg_lora.py`` L2225) did NOT initialize six names that the fall-through
references — five read by the shared final ``metrics.record_step`` (L3938):
``use_cache``, ``cache_eligible``, ``cache_hit``, ``can_confident_skip``,
``m9_cycle_stats``; plus ``is_full_eval_cycle``, which gates the full-eval block
(L4162). All six are assigned only inside the skipped ``if not
dynfreeze_all_frozen:`` block (L2292-3919). The fall-through therefore raised
``UnboundLocalError`` — first on ``use_cache`` in ``record_step``, and (once that
was bound) on ``is_full_eval_cycle`` in the post-record_step tail — so any
enabled-dynfreeze ("Guard") run crashed the *moment it froze the last layer*,
the terminal state the experiment exists to reach.

dynfreeze is disabled in every config on this public mirror, so the crash was
latent (discovered while wiring the resume-state-loss integration test's fault
seam). This test forces the all-frozen path with the real loop code — the
controller is mocked only enough to report all-frozen every cycle — and asserts
the run completes and records the semantically-correct ``tg_lora_cache_built
= False`` (no cache work happened).
"""

import sys
import types
from unittest.mock import MagicMock

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Test-only shim: make the REAL training loop importable on this public mirror
# ---------------------------------------------------------------------------
# Same rationale + design as tests/test_resume_state_integration.py: the public
# mirror excludes the private ``src.data`` pipeline and does not install ``peft``
# (honest known-unavailable deps — see test_cli_help_smoke.py canary xfails).
# train_tg_lora's top-level import pulls both in, so we inject raising-stub
# modules into sys.modules (a test-only shim, NOT a src/ change) to exercise the
# REAL loop code. Every shim'd symbol is mocked in the test deps and never
# called; each raises loudly if reached unmocked rather than returning garbage.
if "src.data" not in sys.modules:
    sys.modules["src.data"] = types.ModuleType("src.data")
if "src.data.build_seed_dataset" not in sys.modules:
    _shim = types.ModuleType("src.data.build_seed_dataset")

    def _load_dataset_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "src.data.load_dataset is private (excluded from this mirror); "
            "tests must mock src.training.train_tg_lora.load_dataset"
        )

    _shim.load_dataset = _load_dataset_unavailable
    sys.modules["src.data.build_seed_dataset"] = _shim

if "src.model.load_model" not in sys.modules:
    _load_model_shim = types.ModuleType("src.model.load_model")

    def _unavailable(_name):
        def _raise(*_args, **_kwargs):
            raise RuntimeError(
                f"src.model.load_model.{_name} needs peft (not installed on this "
                "mirror); train-loop tests must mock "
                f"src.training.train_tg_lora.{_name}"
            )

        return _raise

    for _n in ("apply_lora", "get_input_device", "load_base_model", "load_tokenizer"):
        setattr(_load_model_shim, _n, _unavailable(_n))
    sys.modules["src.model.load_model"] = _load_model_shim

from tests.test_fault_recovery import (  # noqa: E402  (after sys.modules shim)
    _make_run_dir_config,
    _patch_deps,
    _run_with_deps,
)


def _make_all_frozen_config(tmp_path):
    """Run-dir config with the dynfreeze Guard experiment enabled."""
    return OmegaConf.merge(
        _make_run_dir_config(tmp_path),
        OmegaConf.create({"tg_lora": {"dynfreeze_enabled": True}}),
    )


def _all_frozen_controller_mock() -> MagicMock:
    """A dynfreeze controller that reports ALL layers frozen every cycle.

    ``run_cycle`` returning True sends every cycle down the all-frozen skip
    path. The remaining attributes are the ones the final ``record_step``
    reads from the controller's guard-metrics dict
    (``block_size``/``frozen_block``/``_r_A_history``); concrete empty values
    keep that comprehension from touching MagicMock iteration quirks.
    """
    controller = MagicMock()
    controller.run_cycle.return_value = True
    controller.block_size = 0
    controller.frozen_block = []
    controller._r_A_history = {}
    return controller


class TestDynfreezeAllFrozenPath:
    """The all-frozen skip path must initialize every name the shared final
    record_step references — or an enabled Guard run crashes at terminal state."""

    def test_all_frozen_cycle_records_without_crash(self, tmp_path):
        cfg = _make_all_frozen_config(tmp_path)
        deps = _patch_deps(eval_losses=[2.0] * 10, run_dir=tmp_path)

        fake_dynfreeze = _all_frozen_controller_mock()
        deps["src.training.train_tg_lora.DynamicFreezeController"] = MagicMock(
            return_value=fake_dynfreeze
        )

        # Reaches the final record_step on every cycle. Pre-fix this raised
        # UnboundLocalError (use_cache unbound) out of _run_with_deps (not a
        # SystemExit, so not swallowed) → the test errors here.
        mocks = _run_with_deps(cfg, deps)

        metrics = mocks["RunMetrics"].return_value
        assert metrics.record_step.called, (
            "record_step was never called — the all-frozen fall-through crashed "
            "before recording (use_cache / cache_* / can_confident_skip / "
            "m9_cycle_stats unbound in the skip block)"
        )

        # The all-frozen skip path does NO cache work, so every recorded step
        # must carry tg_lora_cache_built=False. This also proves use_cache was
        # initialized (not merely that no exception escaped).
        cache_built_values = [
            call.kwargs.get("tg_lora_cache_built")
            for call in metrics.record_step.call_args_list
            if "tg_lora_cache_built" in call.kwargs
        ]
        assert cache_built_values, (
            "no record_step call carried tg_lora_cache_built — the all-frozen "
            "final record_step did not execute"
        )
        assert all(v is False for v in cache_built_values), (
            f"all-frozen cycles must record tg_lora_cache_built=False (no cache "
            f"work happened), got {cache_built_values}"
        )

        # Sanity: the mock controller's run_cycle actually fired each cycle,
        # so the all-frozen path was genuinely exercised.
        assert fake_dynfreeze.run_cycle.called
