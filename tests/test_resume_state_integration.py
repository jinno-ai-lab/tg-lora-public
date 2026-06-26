"""Integration-level fault-injection resume test for the resume-state-loss axis.

Closes the gap surfaced by the run-feedback review:

    "resume-state-loss is now 9/9 and converging, but the cited verification is
    isolated unit roundtrips per object ... no test proves a real fault-resume
    actually invokes load_training_state and restores ALL persisted fields in
    concert inside the training loop."

Each of the **nine** resume-state-loss sites — ``best_lawa_loss``,
``triggered_target_steps``, ``act_regime_state``, ``efficiency_accounting``,
``psa_state`` (the five the run-feedback review named by example) **plus**
``best_full_eval_loss``/``best_full_eval_perplexity``, ``warmup_released``/
``warmup_cos_consecutive``, ``dynfreeze_state`` and the LAWA snapshot window
``lawa_state`` — already has an ISOLATED round-trip test
(``test_checkpoint.py`` for the ``TrainingState`` layer,
``test_activation_regime.py`` / ``test_psa.py`` / ``test_weight_averaging.py``
for the per-object ``state_dict``/``load_state_dict`` pair). Those prove each
object serializes and deserializes on its own. They do NOT prove that
``train_tg_lora(resume_path=...)`` actually wires the loaded fields back into
the in-loop objects — a deleted restore line, a swapped field, or an over-broad
guard would slip past every isolated round-trip while silently dropping state on
the 9B run's fault/periodic resume. (The dynfreeze site is exactly the class
that bit before: ``119e815`` fixed a ``NameError`` that left dynfreeze state out
of the fault checkpoint — an isolated round-trip stayed green throughout.)

This module runs the REAL resume path end to end — real
``save_training_state`` -> disk -> real ``load_training_state`` -> the real
restore block -> real ``PSAPrior`` / ``ActivationFingerprintTracker`` /
``DynamicFreezeController`` / ``LAWAAverager`` construction + ``load_state_dict``
— with all other heavy deps mocked. It populates ALL nine sites, injects a
numerical-instability fault on the first resumed cycle (so the loop's fault
handler writes a checkpoint from the just-restored in-loop state, before any
cycle body can mutate it), then asserts all nine sites survived in concert
through two independent seams:

  * object fields (``psa_state`` / ``act_regime_state`` / ``dynfreeze_state`` /
    ``lawa_state``): capturing factories that record the in-loop object's state
    the instant ``load_state_dict`` returns — the most direct proof that the
    restore call fired with the right field and produced matching state;
  * plain-local fields (``best_lawa_loss`` / ``triggered_target_steps`` /
    ``efficiency_accounting`` / ``best_full_eval_loss`` / ``warmup_released``):
    the ``TrainingState`` the loop's OWN fault handler hands to
    ``save_training_state``, which re-snapshots the in-loop locals — proving they
    carried through the restore block into live variables.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import torch

# ---------------------------------------------------------------------------
# Test-only shim: make the REAL training loop importable on this public mirror
# ---------------------------------------------------------------------------
# The public mirror intentionally excludes the private ``src.data`` pipeline and
# does not install ``peft`` (an honest known-unavailable dep — see the
# test_cli_help_smoke.py canary xfails). ``train_tg_lora``'s top-level import
# chain pulls both in — ``from src.data.build_seed_dataset import load_dataset``
# directly, and ``from src.model.load_model import (apply_lora, get_input_device,
# load_base_model, load_tokenizer)`` where ``load_model`` itself does
# ``from peft import LoraConfig, get_peft_model`` — so the module is un-importable
# here. That is precisely why the resume-state-loss axis was verified with
# ISOLATED round-trips instead of an end-to-end loop test (and why
# ``test_resume_e2e.py`` errors at collection on this mirror). Every one of these
# symbols is ALWAYS mocked in the test deps (``load_dataset``,
# ``load_base_model``/``apply_lora``/``load_tokenizer``/``get_input_device``), so
# they are NEVER exercised — they only need to resolve so the module imports. We
# inject throwaway modules into ``sys.modules`` (a test-only shim, NOT a ``src/``
# source change that would diverge from upstream) to exercise the REAL resume
# wiring end to end. Each shim raises loudly if ever called unmocked rather than
# returning silent garbage.
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

# Stub ``src.model.load_model``: it is the ONLY module in ``src/`` that imports
# ``peft`` (+ ``transformers.BitsAndBytesConfig``, also unavailable in this
# installed transformers). All four names train_tg_lora re-exports from it are
# mocked in the test deps, so a raising-stub module lets the loop import without
# pulling in peft / mutating the installed transformers. (``iter_lora_params`` /
# ``configure_trainable_lora_scope`` come from ``src.model.lora_utils``, which is
# dependency-free and imports for real — do NOT stub it.)
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

from src.tg_lora.activation_regime import (
    ActivationFingerprintTracker,
    ActivationRegime,
)
from src.tg_lora.dynamic_freeze import DynFreezeState, DynamicFreezeController
from src.tg_lora.psa import PSAPrior
from src.tg_lora.weight_averaging import LAWAAverager
from src.training.trainer_loop import NumericalInstabilityError
from src.utils.checkpoint import (
    TrainingState,
    load_training_state,
    save_training_state,
)
from tests.test_fault_recovery import (
    _make_run_dir_config,
    _make_training_state,
    _patch_deps,
    _run_with_deps_resume,
)

# ---------------------------------------------------------------------------
# Fixtures: a saved TrainingState with ALL five run-wide accumulators populated
# ---------------------------------------------------------------------------


def _populate_psa_prior() -> PSAPrior:
    """A PSAPrior that has accumulated deltas AND extracted priors.

    Mirrors a mid-run prior so the restored object is observably non-empty
    (``history_count >= 2`` is the gate for the run-end ``layer_delta_analysis``;
    non-empty ``priors`` drive production amplification).
    """
    prior = PSAPrior(
        history_length=6,
        gain=0.5,
        update_interval=3,
        warmup_steps=4,
        l2_reg=0.01,
        regime_plateau_gain=0.5,
    )
    for _ in range(2):
        prior.record_delta(
            {
                "layer.0.lora_A": torch.randn(4, 4) * 0.1,
                "layer.0.lora_B": torch.randn(4, 4) * 0.05,
            }
        )
    prior.extract_priors()
    prior.mark_updated(5)
    return prior


def _populate_act_regime_tracker() -> ActivationFingerprintTracker:
    """A regime tracker with a non-trivial cosine series + regime counts."""
    tracker = ActivationFingerprintTracker(window=10, min_history=3)
    tracker._all_cosines.extend([0.97, 0.95, 0.98, 0.96, 0.94, 0.99])
    tracker._counts[ActivationRegime.STABLE] = 5
    tracker._counts[ActivationRegime.CHAOTIC] = 1
    return tracker


def _populate_dynfreeze_state() -> DynFreezeState:
    """A mid-run ``DynFreezeState``: layers {2, 3} frozen since cycle 7, with
    layer 1 released at cycle 5 (mid-cooldown).

    Mirrors a checkpoint taken mid-Guard run so the restored controller is
    observably non-empty (frozen block + a release-cooldown map the §4
    reversible-release path keys on).
    """
    return DynFreezeState(
        frozen_layer_indices=[2, 3],
        r_A_history={2: [0.12, 0.08], 3: [0.05, 0.04]},
        frozen_since_cycle=7,
        prev_A_fro={"layer.2.lora_A": 0.5, "layer.3.lora_A": 0.4},
        median_A=0.3,
        epsilon=1e-6,
        released_at={1: 5},
    )


def _populate_lawa_state() -> dict:
    """A populated LAWA snapshot window (GOAL §3.3): window_size 4, mid-run
    cycle 5, two recorded snapshots.

    ``window_size`` is intentionally 4 (the config constructs the averager at 5)
    so the assertion can prove ``load_state_dict`` overwrote the constructed
    default — not just that an averager exists. The buffer carries distinct
    tensors so a swapped-field / empty-restore restore is unambiguous.
    """
    return {
        "window_size": 4,
        "start_cycle": 2,
        "cycle": 5,
        "recorded_count": 3,
        "buffer": [
            {"layer.0.lora_A": torch.zeros(2, 2), "layer.0.lora_B": torch.ones(2, 2)},
            {"layer.0.lora_A": torch.ones(2, 2), "layer.0.lora_B": torch.zeros(2, 2)},
        ],
    }


def _saved_efficiency_accounting() -> dict:
    """Non-zero run-wide efficiency counters (GOAL §5 / P3) with distinct
    values so a mismatch on any single counter is unambiguous."""
    return {
        "activation_cache_build_count": 3,
        "activation_cache_hit_count": 17,
        "activation_cache_miss_count": 4,
        "pilot_validation_forward_count": 42,
        "post_validation_forward_count": 12,
        "post_extrapolation_eval_count": 8,
        "subspace_zo_attempted_steps_total": 7,
        "subspace_zo_dim1_steps_total": 5,
        "alpha_line_steps_total": 6,
        "future_work_projection_ratios": [0.12, 0.34, 0.56],
        "future_work_internal_pair_count": 9,
    }


def _build_saved_state(tmp_path, expected):
    """Build a realistic TrainingState (cycle_offset=3) carrying all 5 fields.

    Reuses ``_make_training_state`` for the legacy fields, then attaches the
    five run-wide accumulators and records the ground-truth values the test
    will assert against in ``expected``.
    """
    psa_prior = _populate_psa_prior()
    act_tracker = _populate_act_regime_tracker()

    expected["psa_history_count"] = psa_prior.history_count
    expected["psa_prior_count"] = len(psa_prior.priors)
    expected["psa_last_update_step"] = psa_prior._last_update_step
    expected["psa_state"] = psa_prior.state_dict()

    expected["act_cosine_count"] = len(act_tracker._all_cosines)
    expected["act_stable_fraction"] = act_tracker.stable_fraction
    expected["act_counts"] = dict(act_tracker._counts)
    expected["act_regime_state"] = act_tracker.state_dict()

    expected["best_lawa_loss"] = 1.23
    expected["triggered_target_steps"] = [250, 500]
    expected["efficiency_accounting"] = _saved_efficiency_accounting()

    # The four sites the original 5-field capstone did not exercise end to end.
    expected["best_full_eval_loss"] = 0.879
    expected["best_full_eval_perplexity"] = 2.34
    expected["warmup_released"] = True
    expected["warmup_cos_consecutive"] = 3
    expected["dynfreeze_state"] = _populate_dynfreeze_state()
    expected["dynfreeze_frozen_layers"] = {2, 3}
    expected["dynfreeze_frozen_since"] = 7
    expected["dynfreeze_released_at"] = {1: 5}
    expected["lawa_state"] = _populate_lawa_state()
    expected["lawa_window_size"] = 4
    expected["lawa_buffer_len"] = 2
    expected["lawa_recorded_count"] = 3

    base = _make_training_state()  # cycle_offset=3, realistic legacy fields
    base.best_lawa_loss = expected["best_lawa_loss"]
    base.triggered_target_steps = list(expected["triggered_target_steps"])
    base.act_regime_state = expected["act_regime_state"]
    base.efficiency_accounting = expected["efficiency_accounting"]
    base.psa_state = expected["psa_state"]
    base.best_full_eval_loss = expected["best_full_eval_loss"]
    base.best_full_eval_perplexity = expected["best_full_eval_perplexity"]
    base.warmup_released = expected["warmup_released"]
    base.warmup_cos_consecutive = expected["warmup_cos_consecutive"]
    base.dynfreeze_state = expected["dynfreeze_state"]
    base.lawa_state = expected["lawa_state"]

    state_path = tmp_path / "training_state.pt"
    save_training_state(base, state_path)
    return state_path


# ---------------------------------------------------------------------------
# Fixtures: a mock model that exposes a detectable decoder-layer container
# ---------------------------------------------------------------------------


class _ModelWithDecoderLayers(torch.nn.Module):
    """LoRA mock model that ALSO exposes a detectable decoder-layer container.

    ``train_tg_lora``'s activation-regime setup locates the last decoder layer
    via ``for _name, _mod in model.named_modules(): if hasattr(_mod, "layers")
    and hasattr(_mod, "__len__"): _target_layer = _mod[-1]``. The shared
    ``_LoRAMockModel`` has no such container, so ``activation_regime_enabled``
    would silently set ``act_regime_tracker = None`` (line ~1676) and the
    ``act_regime_state`` restore would never fire — defeating this test. This
    model mirrors ``_LoRAMockModel``'s LoRA-param layout for ``snapshot_lora``
    while ALSO satisfying the layer-detection probe so the in-loop tracker is
    built and restored.
    """

    def __init__(self, num_layers: int = 4, hidden: int = 8):
        super().__init__()
        # The decoder-layer container the activation-regime probe keys on.
        # ``__len__`` / ``__getitem__`` live on the class (Python dunder
        # protocol resolves them on the type, not the instance).
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, hidden) for _ in range(num_layers)]
        )
        for i in range(num_layers):
            setattr(
                self,
                f"layers_{i}_lora_A",
                torch.nn.Parameter(torch.randn(hidden, hidden) * 0.01),
            )
            setattr(
                self,
                f"layers_{i}_lora_B",
                torch.nn.Parameter(torch.randn(hidden, hidden) * 0.01),
            )
        self.save_pretrained = MagicMock()

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def parameters(self):
        for p in super().parameters():
            p.requires_grad = True
            yield p

    def named_parameters(self, **kwargs):
        for name, p in super().named_parameters(**kwargs):
            yield name, p

    def train(self, mode: bool = True):
        return self

    def forward(self, *args, **kwargs):  # never reached: forward_backward mocked
        out = MagicMock()
        out.loss = torch.tensor(2.0, requires_grad=True)
        return out


# ---------------------------------------------------------------------------
# Capture seams
# ---------------------------------------------------------------------------


def _capturing_factory(orig_cls, capture_key, captured: dict):
    """Return a factory that builds a real ``orig_cls`` but records the instance
    the instant ``load_state_dict`` returns — proving the restore call fired
    and capturing the resulting in-loop state for direct comparison."""

    def factory(*args, **kwargs):
        obj = orig_cls(*args, **kwargs)
        orig_load = obj.load_state_dict

        def wrapped_load(state):
            orig_load(state)
            captured[capture_key] = obj

        obj.load_state_dict = wrapped_load
        return obj

    return factory


def _capturing_save(real_save, sink: list):
    """Wrap save_training_state to record the TrainingState the loop persists,
    then delegate to the real serializer so the on-disk file is still written."""

    def wrapped(ts: TrainingState, path):
        sink.append(ts)
        return real_save(ts, path)

    return wrapped


def _make_resume_state_config(tmp_path):
    """Run-dir config with PSA + activation-regime + dynfreeze + LAWA ON so the
    in-loop prior / tracker / controller / averager are built and their restore
    paths actually fire."""
    cfg = _make_run_dir_config(tmp_path)
    # OmegaConf merges these into the existing tg_lora block.
    from omegaconf import OmegaConf

    return OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "tg_lora": {
                    "enable_psa": True,
                    "psa_history_length": 6,
                    "psa_update_interval": 3,
                    "psa_warmup_steps": 4,
                    "activation_regime_enabled": True,
                    "activation_regime_window": 10,
                    "activation_regime_min_history": 3,
                    # Guard reversible-freeze controller — built so the in-loop
                    # controller exists for the dynfreeze_state restore to fire.
                    "dynfreeze_enabled": True,
                    # LAWA mandatory baseline (GOAL §3.3) — built so the in-loop
                    # averager exists for the lawa_state window restore to fire.
                    # window_size 5 intentionally != the saved 4, so the
                    # assertion proves load_state_dict overwrote the default.
                    "enable_lawa": True,
                    "lawa_window_size": 5,
                    "lawa_start_cycle": 10,
                }
            }
        ),
    )


def _run_fault_resume(tmp_path, state_path):
    """Resume from *state_path*, faulting on the first resumed cycle, returning
    (captured_in_loop_objects, captured_saved_states)."""
    cfg = _make_resume_state_config(tmp_path)
    deps = _patch_deps(eval_losses=[2.0, 1.5] * 10, run_dir=tmp_path)

    # Decoder-layer model so the activation-regime tracker is built (not None).
    model = _ModelWithDecoderLayers()
    deps["src.training.train_tg_lora.load_base_model"] = MagicMock(
        return_value=model
    )
    deps["src.training.train_tg_lora.apply_lora"] = MagicMock(return_value=model)

    # Capture the in-loop objects right after their load_state_dict restore.
    captured_objs: dict = {}
    deps["src.training.train_tg_lora.PSAPrior"] = _capturing_factory(
        PSAPrior, "psa_prior", captured_objs
    )
    deps["src.training.train_tg_lora.ActivationFingerprintTracker"] = (
        _capturing_factory(
            ActivationFingerprintTracker, "act_regime_tracker", captured_objs
        )
    )
    deps["src.training.train_tg_lora.DynamicFreezeController"] = _capturing_factory(
        DynamicFreezeController, "dynfreeze", captured_objs
    )
    deps["src.training.train_tg_lora.LAWAAverager"] = _capturing_factory(
        LAWAAverager, "lawa_averager", captured_objs
    )

    # Capture the fault-checkpoint TrainingState the loop's own handler writes.
    captured_saves: list = []
    deps["src.training.train_tg_lora.save_training_state"] = _capturing_save(
        save_training_state, captured_saves
    )

    # Fault on the first resumed cycle's dynfreeze.run_cycle — the FIRST action
    # after the restore block (train_tg_lora.py:1999), before any cycle body can
    # mutate the restored locals. This seam is downstream-stable in a way the
    # pilot's forward_backward is NOT: forward_backward only runs when the
    # dynfreeze-gated pilot is not skipped, so a mutation that changes the
    # controller's frozen block (e.g. disabling its restore, which lets the fresh
    # controller freeze-all on cycle 1) would skip the pilot and prevent the
    # fault from firing. Faulting at run_cycle makes the fault deterministic
    # w.r.t. all nine restore sites, so each site's assertion is reachable when
    # its restore line is mutated.
    fault = NumericalInstabilityError("Loss is nan (non-finite)")

    # The loop ends a fault run by raising SystemExit(2) (train_tg_lora.py:4556).
    # ``_run_with_deps_resume`` does not swallow it (unlike ``_run_with_deps``),
    # so the fault path is observed here. Re-raise a non-fault exit code so a
    # clean finish (which would mean the fault never fired) is not masked.
    with patch.object(DynamicFreezeController, "run_cycle", side_effect=fault):
        try:
            _run_with_deps_resume(cfg, deps, str(state_path))
        except SystemExit as exc:
            if exc.code not in (2,):
                raise
    return captured_objs, captured_saves


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResumeStateIntegration:
    """One integration-level fault-injection resume asserting ALL nine
    resume-state-loss sites survive ``load_training_state`` -> restore block."""

    def test_all_run_wide_state_survives_fault_resume(self, tmp_path):
        expected: dict = {}
        state_path = _build_saved_state(tmp_path, expected)

        # Sanity: the on-disk checkpoint round-trips ALL nine sites.
        loaded = load_training_state(state_path)
        assert loaded.cycle_offset == 3
        assert loaded.best_lawa_loss == expected["best_lawa_loss"]
        assert set(loaded.triggered_target_steps) == {250, 500}
        assert loaded.psa_state is not None
        assert loaded.act_regime_state is not None
        assert loaded.efficiency_accounting is not None
        # The four sites the 5-field capstone omitted.
        assert loaded.best_full_eval_loss == expected["best_full_eval_loss"]
        assert (
            loaded.best_full_eval_perplexity == expected["best_full_eval_perplexity"]
        )
        assert loaded.warmup_released is True
        assert loaded.warmup_cos_consecutive == expected["warmup_cos_consecutive"]
        assert loaded.dynfreeze_state is not None
        assert sorted(loaded.dynfreeze_state.frozen_layer_indices) == [2, 3]
        assert loaded.lawa_state is not None
        assert loaded.lawa_state["window_size"] == expected["lawa_window_size"]

        captured_objs, captured_saves = _run_fault_resume(tmp_path, state_path)

        # The loop's fault handler must have persisted a checkpoint from the
        # resumed run — exactly one (the fault fires on the first resumed
        # cycle's pilot, before any periodic save).
        assert len(captured_saves) == 1, (
            f"Expected exactly one fault-checkpoint save, got {len(captured_saves)}"
        )
        fault_ts = captured_saves[0]
        assert fault_ts.cycle_offset == loaded.cycle_offset, (
            "Fault checkpoint must reflect the resumed cycle, not a fresh run"
        )

        # ---- psa_state (object, via load_state_dict) ----
        restored_psa = captured_objs.get("psa_prior")
        assert restored_psa is not None, (
            "psa_prior.load_state_dict was never called on resume — the PSA "
            "subspace-prior restore wiring is broken"
        )
        assert restored_psa.history_count == expected["psa_history_count"]
        assert len(restored_psa.priors) == expected["psa_prior_count"]
        assert restored_psa._last_update_step == expected["psa_last_update_step"]
        # The in-loop prior re-snapshotted by the loop's own save must match too.
        assert fault_ts.psa_state is not None
        assert len(fault_ts.psa_state["delta_history"]) == expected[
            "psa_history_count"
        ]

        # ---- act_regime_state (object, via load_state_dict) ----
        restored_act = captured_objs.get("act_regime_tracker")
        assert restored_act is not None, (
            "act_regime_tracker.load_state_dict was never called on resume — "
            "the activation-regime inventory restore wiring is broken (or the "
            "model exposed no detectable decoder layer)"
        )
        assert len(restored_act._all_cosines) == expected["act_cosine_count"]
        assert restored_act._counts == expected["act_counts"]
        assert restored_act.stable_fraction == expected["act_stable_fraction"]
        assert fault_ts.act_regime_state is not None
        assert len(fault_ts.act_regime_state["all_cosines"]) == expected[
            "act_cosine_count"
        ]

        # ---- best_lawa_loss (plain local, via fault-checkpoint re-snapshot) ----
        assert fault_ts.best_lawa_loss == expected["best_lawa_loss"], (
            "best_lawa_loss did not survive resume — the LAWA headline tracker "
            "restore wiring is broken (resumed at inf)"
        )

        # ---- triggered_target_steps (plain local) ----
        assert set(fault_ts.triggered_target_steps) == set(
            expected["triggered_target_steps"]
        ), (
            "triggered_target_steps did not survive resume — the linearity-budget "
            "target-step set restore wiring is broken (resumed empty)"
        )

        # ---- efficiency_accounting (plain locals) ----
        restored_eff = fault_ts.efficiency_accounting
        assert restored_eff is not None, (
            "efficiency_accounting did not survive resume — the run-wide cost "
            "counter restore wiring is broken (resumed empty)"
        )
        for key, value in expected["efficiency_accounting"].items():
            assert key in restored_eff, (
                f"efficiency counter '{key}' missing from resumed state"
            )
            assert restored_eff[key] == value, (
                f"efficiency counter '{key}' did not survive resume: "
                f"expected {value!r}, got {restored_eff[key]!r}"
            )

        # ---- dynfreeze_state (object, via load_state_dict) ----
        restored_dynfreeze = captured_objs.get("dynfreeze")
        assert restored_dynfreeze is not None, (
            "dynfreeze.load_state_dict was never called on resume — the "
            "reversible-freeze controller restore wiring is broken (or "
            "dynfreeze was not enabled, so the controller was never built)"
        )
        assert restored_dynfreeze.frozen_layer_indices == expected[
            "dynfreeze_frozen_layers"
        ], "dynfreeze frozen block did not survive resume"
        assert restored_dynfreeze._frozen_since_cycle == expected[
            "dynfreeze_frozen_since"
        ], "dynfreeze frozen_since_cycle did not survive resume"
        assert restored_dynfreeze._released_at == expected[
            "dynfreeze_released_at"
        ], "dynfreeze release-cooldown map did not survive resume"
        # The in-loop controller re-snapshotted by the loop's own save matches.
        assert fault_ts.dynfreeze_state is not None
        assert sorted(fault_ts.dynfreeze_state.frozen_layer_indices) == [2, 3]

        # ---- best_full_eval_loss / perplexity (plain locals) ----
        assert fault_ts.best_full_eval_loss == expected["best_full_eval_loss"], (
            "best_full_eval_loss did not survive resume — the best-model "
            "save-gate tracker restore wiring is broken (resumed at inf, so the "
            "first post-resume full eval would clobber the genuine best_model)"
        )
        assert fault_ts.best_full_eval_perplexity == expected[
            "best_full_eval_perplexity"
        ], "best_full_eval_perplexity did not survive resume"

        # ---- warmup_released / warmup_cos_consecutive (plain locals) ----
        assert fault_ts.warmup_released is expected["warmup_released"], (
            "warmup_released did not survive resume — the two-phase warmup gate "
            "restore wiring is broken (resumed False, silently dropping a "
            "mid-production checkpoint back into the pilot-only warmup phase)"
        )
        assert fault_ts.warmup_cos_consecutive == expected["warmup_cos_consecutive"], (
            "warmup_cos_consecutive did not survive resume"
        )

        # ---- lawa_state window (object, via load_state_dict) ----
        restored_lawa = captured_objs.get("lawa_averager")
        assert restored_lawa is not None, (
            "lawa_averager.load_state_dict was never called on resume — the "
            "LAWA snapshot-window restore wiring is broken (or LAWA was not "
            "enabled, so the averager was never built)"
        )
        assert restored_lawa.window_size == expected["lawa_window_size"], (
            "LAWA window_size was not restored — the saved window must overwrite "
            "the config-constructed default (proving load_state_dict fired)"
        )
        assert restored_lawa.count == expected["lawa_buffer_len"], (
            "LAWA snapshot buffer did not survive resume"
        )
        assert restored_lawa._recorded_count == expected["lawa_recorded_count"], (
            "LAWA recorded_count did not survive resume"
        )
        assert fault_ts.lawa_state is not None
        assert len(fault_ts.lawa_state["buffer"]) == expected["lawa_buffer_len"]
