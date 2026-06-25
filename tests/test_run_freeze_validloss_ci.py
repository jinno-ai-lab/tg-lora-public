"""Tests for ``scripts/run_freeze_validloss_ci.py`` — the Category-C attack.

This is the run that finally feeds :func:`surrogate_valid_loss_ci` numbers that
came out of an *actual* training run rather than constructed constants (see the
script docstring). The suite guards:

* **Import health + ``--help``** — the CLI is launchable as ``-m`` (the canary
  contract every ``scripts.run_*`` CLI in this repo keeps).
* **The arm is a real training run.** :func:`arm_valid_loss` returns a finite
  value below the uniform-init loss (the proxy genuinely learns, not a stub),
  is reproducible for a fixed ``(order, seed)`` (every RNG is locally seeded),
  and is not a constant across seeds.
* **The CI is wired correctly.** :func:`run_ci` deposits one real valid_loss
  sample per arm, the verdict is one of the three valid §4 labels, and the
  verdict is *self-consistent* with the bootstrap CI bounds (re-derived from
  ``lower``/``upper``) — proving the harness hands the samples to the right
  statistic rather than returning a label by another route.
* **Honest proxy-scale labeling.** The report and JSON carry the
  ``PROXY_SCALE`` caveat a reader must see before citing the verdict.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest

from src.tg_lora.freeze_surrogate_gate import SURPASSES, TIES, UNDERSHOOTS

# Tiny but non-thin (>= 3 seeds/arm, see MIN_SAMPLE_FOR_BOOTSTRAP) budget for a
# fast, deterministic CPU check. The make target / real GPU run use the larger
# converged-regime defaults; this only exercises the wiring.
_TINY_ARM = dict(total=15, warmup=4, depth=2)
_TINY = dict(total=15, warmup=4, depth=2, n_candidate=3, n_surrogate=3)
_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Import health + --help
# ---------------------------------------------------------------------------


class TestImportHealth:
    def test_module_imports_successfully(self):
        mod = importlib.import_module("scripts.run_freeze_validloss_ci")
        for attr in (
            "main",
            "build_parser",
            "run_ci",
            "arm_valid_loss",
            "format_report",
            "result_to_json",
            "resolve_device",
            "output_first_order",
        ):
            assert hasattr(mod, attr), f"missing {attr}"

    def test_output_first_order_descends_from_output_side(self):
        from scripts.run_freeze_validloss_ci import output_first_order

        assert output_first_order(6) == (5, 4, 3, 2, 1, 0)
        assert output_first_order(4) == (3, 2, 1, 0)


class TestCLIHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_freeze_validloss_ci", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "valid_loss" in result.stdout.lower()
        # The §4 control (random-order surrogate) and the candidate must surface.
        assert "surrogate" in result.stdout.lower()
        assert "candidate" in result.stdout.lower()


# ---------------------------------------------------------------------------
# The arm is a real, reproducible training run
# ---------------------------------------------------------------------------


class TestArmIsRealTraining:
    def test_arm_returns_finite_learned_loss(self):
        from scripts.run_freeze_validloss_ci import arm_valid_loss, output_first_order

        v = arm_valid_loss(
            output_first_order(6), seed=0, device=_DEVICE, num_layers=6, **_TINY_ARM
        )
        # Finite, and the proxy genuinely learned: well below the ~log(32)=3.47
        # uniform-init loss (the invivo fixture descends to ~0.4 over a fuller
        # budget; this tiny budget still lands far below uniform).
        assert v == pytest.approx(v)  # finite (not nan/inf)
        assert v < 2.5, f"arm did not learn: valid_loss={v}"

    def test_arm_is_reproducible_for_fixed_order_and_seed(self):
        from scripts.run_freeze_validloss_ci import arm_valid_loss, output_first_order

        order = output_first_order(6)
        a = arm_valid_loss(order, seed=7, device=_DEVICE, num_layers=6, **_TINY_ARM)
        b = arm_valid_loss(order, seed=7, device=_DEVICE, num_layers=6, **_TINY_ARM)
        # Every RNG (torch init, batch generator, the CI's numpy seed) is locally
        # seeded, so an arm is bit-reproducible on a fixed device.
        assert a == b

    def test_arm_varies_across_seeds_not_a_constant(self):
        from scripts.run_freeze_validloss_ci import arm_valid_loss, output_first_order

        order = output_first_order(6)
        vals = [
            arm_valid_loss(order, seed=s, device=_DEVICE, num_layers=6, **_TINY_ARM)
            for s in (0, 1, 2)
        ]
        assert all(v == pytest.approx(v) for v in vals)  # all finite
        # The value is computed from a real run, not hardcoded: distinct seeds
        # produce a genuine spread (at least two differ).
        assert len(set(round(v, 6) for v in vals)) >= 2


# ---------------------------------------------------------------------------
# run_ci deposits real samples and wires them to the correct statistic
# ---------------------------------------------------------------------------


class TestRunCI:
    def test_run_ci_deposits_one_real_sample_per_arm_and_is_non_thin(self):
        from scripts.run_freeze_validloss_ci import run_ci

        r = run_ci(device=_DEVICE, num_layers=6, **_TINY)
        cand = r["candidate_losses"]
        surr = r["surrogate_losses"]
        assert len(cand) == _TINY["n_candidate"]
        assert len(surr) == _TINY["n_surrogate"]
        # Real samples: finite, and the arms actually learned (below uniform init).
        assert all(v == pytest.approx(v) for v in cand + surr)
        assert max(cand + surr) < 2.5
        # Non-thin: both arms meet MIN_SAMPLE_FOR_BOOTSTRAP (=3).
        ci = r["ci"]
        assert ci.is_thin_evidence is False

    def test_verdict_is_valid_label_and_self_consistent_with_ci_bounds(self):
        from scripts.run_freeze_validloss_ci import run_ci

        r = run_ci(device=_DEVICE, num_layers=6, **_TINY)
        ci = r["ci"]
        # The verdict is one of the three §4 labels the structural gate emits.
        assert ci.significance_verdict in {SURPASSES, TIES, UNDERSHOOTS}
        # Self-consistency — re-derive the verdict from the CI bounds: this proves
        # the harness handed the real samples to surrogate_valid_loss_ci (which
        # sets lower/upper) rather than producing the label another way.
        if ci.significance_verdict == SURPASSES:
            assert ci.lower > 0.0
        elif ci.significance_verdict == UNDERSHOOTS:
            assert ci.upper < 0.0
        else:  # TIES
            assert ci.lower <= 0.0 <= ci.upper
        # The point estimate is the observed difference of means, in valid_loss units.
        assert ci.point_improvement == pytest.approx(
            ci.surrogate_mean - ci.candidate_mean
        )

    def test_run_ci_is_reproducible_by_base_seed(self):
        from scripts.run_freeze_validloss_ci import run_ci

        a = run_ci(device=_DEVICE, base_seed=42, num_layers=6, **_TINY)
        b = run_ci(device=_DEVICE, base_seed=42, num_layers=6, **_TINY)
        assert a["candidate_losses"] == b["candidate_losses"]
        assert a["surrogate_losses"] == b["surrogate_losses"]
        assert a["ci"].significance_verdict == b["ci"].significance_verdict
        assert a["ci"].lower == b["ci"].lower and a["ci"].upper == b["ci"].upper


# ---------------------------------------------------------------------------
# Honest proxy-scale labeling in both renderings
# ---------------------------------------------------------------------------


class TestHonestProxyScaleLabeling:
    def test_report_carries_proxy_scale_caveat(self):
        from scripts.run_freeze_validloss_ci import format_report, run_ci

        r = run_ci(device=_DEVICE, num_layers=6, **_TINY)
        text = format_report(r)
        assert "PROXY_SCALE" in text
        # The caveat tells a reader not to cite it as a target-scale result.
        assert "9B target" in text
        assert "proxy_scale=True" in text

    def test_json_carries_proxy_scale_flag(self):
        from scripts.run_freeze_validloss_ci import result_to_json, run_ci

        r = run_ci(device=_DEVICE, num_layers=6, **_TINY)
        payload = result_to_json(r)
        assert payload["proxy_scale"] is True
        # The JSON is the evidence artifact a target-scale run would overwrite.
        parsed = json.loads(json.dumps(payload))
        assert parsed["verdict"] in {SURPASSES, TIES, UNDERSHOOTS}
        assert len(parsed["candidate_losses"]) == _TINY["n_candidate"]
        assert len(parsed["surrogate_losses"]) == _TINY["n_surrogate"]


class TestTargetScaleParam:
    """``run_ci`` carries a caller-supplied ``proxy_scale`` — not a hardcoded True.

    The MS-PF2 Cat-C contract is that a target-scale source deposits samples in
    the *same* schema and the scale label carries through with no code change, so
    the scale must be a value the caller sets, not a magic ``True`` baked into the
    result dict. The default stays proxy (byte-identical to the recorded
    fixtures); ``proxy_scale=False`` threads through to the result, the JSON, and
    the report's TARGET_SCALE note. This is the generator half of the drop-in
    path the replay-side suite validates on a committed ``proxy_scale: false``
    fixture.
    """

    def test_proxy_scale_false_threads_to_result_and_json(self):
        from scripts.run_freeze_validloss_ci import run_ci, result_to_json

        r = run_ci(device=_DEVICE, num_layers=6, proxy_scale=False, **_TINY)
        assert r["proxy_scale"] is False
        payload = result_to_json(r)
        assert payload["proxy_scale"] is False

    def test_proxy_scale_default_is_true_byte_identical(self):
        from scripts.run_freeze_validloss_ci import run_ci

        # Default stays proxy-scale — the recorded fixtures (proxy_scale=True)
        # are unchanged, so nothing already committed regresses.
        r = run_ci(device=_DEVICE, num_layers=6, **_TINY)
        assert r["proxy_scale"] is True

    def test_report_renders_target_scale_note_when_false(self):
        from scripts.run_freeze_validloss_ci import format_report, run_ci

        r = run_ci(device=_DEVICE, num_layers=6, proxy_scale=False, **_TINY)
        text = format_report(r)
        # The proxy caveat is replaced by the target-scale note; the header line
        # reports proxy_scale=False.
        assert "proxy_scale=False" in text
        assert "TARGET_SCALE" in text
        assert "PROXY_SCALE" not in text


# ---------------------------------------------------------------------------
# Heterogeneous positive control + generalize conclusive-TIES task
# (the two apparatus-validation axes added to the Category-C attack)
# ---------------------------------------------------------------------------


class TestHeterogeneousRanks:
    def test_geometric_schedule_rises_toward_output(self):
        from scripts.run_freeze_validloss_ci import heterogeneous_ranks, HIDDEN

        r = heterogeneous_ranks(6, HIDDEN)
        # Strictly rising toward the output side (more adapter capacity near the
        # output) — the faithful proxy of GOAL §1.5/§8 non-uniform per-layer cost.
        assert r == (1, 2, 4, 7, 13, 24)
        assert all(r[i] < r[i + 1] for i in range(len(r) - 1))

    def test_single_layer_collapses_to_full_rank(self):
        from scripts.run_freeze_validloss_ci import heterogeneous_ranks, HIDDEN

        assert heterogeneous_ranks(1, HIDDEN) == (HIDDEN,)


class TestRunCIHeterogeneousAndGeneralize:
    def test_heterogeneous_records_per_layer_ranks_not_uniform(self):
        from scripts.run_freeze_validloss_ci import (
            HETEROGENEOUS,
            HOMOGENEOUS,
            HIDDEN,
            run_ci,
        )

        het = run_ci(
            device=_DEVICE, architecture=HETEROGENEOUS, num_layers=6, **_TINY
        )
        hom = run_ci(
            device=_DEVICE, architecture=HOMOGENEOUS, num_layers=6, **_TINY
        )
        # Heterogeneous deposits the geometric per-layer rank schedule; the
        # homogeneous default records the full HIDDEN rank on every layer.
        assert het["architecture"] == HETEROGENEOUS
        assert het["ranks"] == [1, 2, 4, 7, 13, 24]
        assert hom["architecture"] == HOMOGENEOUS
        assert hom["ranks"] == [HIDDEN] * 6
        # Both deposit real samples and a valid verdict on either stack.
        for r in (het, hom):
            assert len(r["candidate_losses"]) == _TINY["n_candidate"]
            assert len(r["surrogate_losses"]) == _TINY["n_surrogate"]
            assert r["ci"].significance_verdict in {SURPASSES, TIES, UNDERSHOOTS}

    def test_generalize_task_is_labeled_and_deposits_real_samples(self):
        from scripts.run_freeze_validloss_ci import TASK_GENERALIZE, run_ci

        r = run_ci(device=_DEVICE, task=TASK_GENERALIZE, num_layers=6, **_TINY)
        assert r["task"] == TASK_GENERALIZE
        samples = r["candidate_losses"] + r["surrogate_losses"]
        assert all(v == pytest.approx(v) for v in samples)  # all finite

    def test_json_carries_architecture_ranks_and_task(self):
        from scripts.run_freeze_validloss_ci import (
            HETEROGENEOUS,
            TASK_GENERALIZE,
            result_to_json,
            run_ci,
        )

        r = run_ci(
            device=_DEVICE,
            architecture=HETEROGENEOUS,
            task=TASK_GENERALIZE,
            num_layers=6,
            **_TINY,
        )
        payload = result_to_json(r)
        assert payload["architecture"] == HETEROGENEOUS
        assert payload["ranks"] == [1, 2, 4, 7, 13, 24]
        assert payload["task"] == TASK_GENERALIZE

    def test_cli_exposes_new_flags(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_freeze_validloss_ci", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "heterogeneous" in result.stdout
        assert "generalize" in result.stdout


class TestTeacherConfidence:
    """The generalization target must be a *confident* function, not noise.

    A near-uniform teacher (softmax entropy ~= log(VOCAB)) has a near-random
    argmax, so its labels are unlearnable noise and any verdict is the trivial
    "student couldn't learn" kind. The calibrated teacher must sit well below
    uniform — the apparatus-validation invariant that makes a generalize-task
    TIES conclusive rather than vacuous.
    """

    def test_teacher_entropy_is_well_below_uniform(self):
        import torch
        import torch.nn.functional as F

        from scripts.run_freeze_validloss_ci import (
            HIDDEN,
            NUM_LAYERS,
            VOCAB,
            _frozen_teacher,
            make_generalize_task,
        )

        teacher = _frozen_teacher(NUM_LAYERS, HIDDEN, "cpu")
        train, _ = make_generalize_task(0, teacher=teacher)
        logits = teacher(
            input_ids=train[0]["input_ids"],
            attention_mask=train[0]["attention_mask"],
        ).logits
        p = F.softmax(logits, dim=-1)
        entropy = (-(p * torch.log(p + 1e-9)).sum(-1).mean()).item()
        uniform = float(torch.log(torch.tensor(float(VOCAB))))
        # Confident with margin: the calibration lands near ~1.1 nats vs ~3.47
        # uniform; assert < 2.0 so the invariant is robust to float drift.
        assert entropy < 2.0, (
            f"teacher too uniform: entropy={entropy:.3f} vs uniform={uniform:.3f}"
        )

    def test_teacher_is_frozen_and_in_eval_mode(self):
        from scripts.run_freeze_validloss_ci import HIDDEN, NUM_LAYERS, _frozen_teacher

        teacher = _frozen_teacher(NUM_LAYERS, HIDDEN, "cpu")
        assert teacher.training is False
        assert all(not p.requires_grad for p in teacher.parameters())


class TestGeneralizeArmLearns:
    """The conclusive-TIES precondition: the student LEARNS the held-out task.

    At the full budget the student reaches held-out valid_loss well below the
    uniform ~3.47 — it generalizes the teacher's function, not memorizes. That
    is what makes a generalize-task TIES conclusive: the model demonstrably
    learned, yet order still did not help. A full CI is a 10-arm run; one arm
    is enough to prove the learnability invariant and stays fast.
    """

    def test_single_generalize_arm_generalizes_below_uniform(self):
        from scripts.run_freeze_validloss_ci import arm_valid_loss, output_first_order

        v = arm_valid_loss(
            output_first_order(6),
            seed=0,
            device=_DEVICE,
            num_layers=6,
            total=60,
            warmup=45,
            depth=3,
            task="generalize",
        )
        assert v == pytest.approx(v)  # finite
        # Well below uniform: the student generalizes the teacher's function,
        # the precondition for a conclusive (not trivial) TIES verdict.
        assert v < 3.0, f"generalize arm did not learn: held-out valid_loss={v}"


# ---------------------------------------------------------------------------
# Apparatus drift sentinel — guard the committed GPU fixture's reproducibility
# ---------------------------------------------------------------------------


class TestApparatusDriftSentinel:
    """Pin the apparatus's deterministic output so silent drift fails CI.

    The committed Category-C GPU recording
    (``tests/fixtures/freeze_validloss_generalize_proxy.json``) is only checked
    for *faithfulness* — that its frozen floats replay to the verdict it records
    (``tests/test_replay_freeze_validloss_ci.py::TestFixtureFaithfulness``). That
    test passes even if the apparatus rots, because the floats in the file never
    change. What no CI check guarded is that the documented regeneration command
    (``make freeze-validloss-ci-generalize``) still reproduces that recording —
    verified manually to reproduce **bit-for-bit** on the RTX 3060 (2026-06-26:
    verdict TIES, every mean / CI bound / loss identical to the fixture), but
    that was an assertion, not an enforced contract.

    This sentinel closes that gap the cheap way. The apparatus is bit-deterministic
    (``test_run_ci_is_reproducible_by_base_seed`` proves exact within-session
    reproducibility), so pinning its tiny-budget CPU ``generalize`` output makes
    any drift in the student-affecting constants/logic the GPU run depends on
    (``TEACHER_BASE_STD`` / ``TEACHER_SEED`` / ``LR`` / ``make_generalize_task`` /
    ``arm_valid_loss``) fail CI — mutation-verified: nudging each of those three
    constants moves the pinned floats. A failure here means the GPU fixture is now
    stale: regenerate it (``make freeze-validloss-ci-generalize``) and re-pin
    these golden values — re-validating the evidence chain rather than letting
    it rot silently.

    Scope is honest about what it does NOT catch. A constant whose effect is
    argmax-invariant on the student leaves the loss — and this sentinel — unmoved.
    ``TEACHER_HEAD_SCALE`` is one: it sharpens the teacher's softmax (lowering its
    entropy, the invariant ``TestTeacherConfidence`` pins) but does not change the
    argmax *labels* the student trains on, so the student's loss is identical
    across its values. The sentinel therefore guards the training-affecting
    constants; the calibration-affecting ones are covered by the teacher-entropy
    test. It is a drift sentinel, NOT a research claim: the tiny budget barely learns
    (valid_loss ~3.1 vs uniform ~3.47) and is chosen for speed; only the
    *stability* of the output across commits is asserted, not the value as a
    §4 result.
    """

    # Golden output of run_ci(generalize, tiny CPU budget), captured from a real
    # deterministic run. Every RNG is locally seeded, so this is bit-stable on a
    # fixed torch/CPU — exactly the determinism the GPU recording relies on.
    _GOLDEN_CANDIDATE = (3.1415646076202393, 2.9830827713012695, 3.1431686878204346)
    _GOLDEN_SURROGATE = (3.3331193923950195, 2.9940176010131836, 3.2981650829315186)

    def test_generalize_tiny_output_is_pinned(self):
        from scripts.run_freeze_validloss_ci import TASK_GENERALIZE, run_ci

        r = run_ci(device=_DEVICE, task=TASK_GENERALIZE, num_layers=6, **_TINY)
        # Exact equality: the apparatus is bit-deterministic (the reproducibility
        # test asserts a == b within a session), so any change in the pinned
        # constants/logic moves these floats and fails the sentinel — flagging
        # that the GPU fixture must be regenerated.
        assert tuple(r["candidate_losses"]) == self._GOLDEN_CANDIDATE
        assert tuple(r["surrogate_losses"]) == self._GOLDEN_SURROGATE
        assert r["ci"].significance_verdict == TIES
