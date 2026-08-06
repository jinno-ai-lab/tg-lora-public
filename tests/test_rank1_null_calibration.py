"""Behavior-lock: rank-1 z-score diagnostic is calibrated on the iid null.

Responds to AI_HUB_MAKE_RUN_FEEDBACK's falsifiable claim that the
Marchenko-Pastur null in ``marchenko_pastur_expected_rank1`` (the comment vs.
implementation mismatch — comment claims ``(1 + sqrt(n/p))^2``, implementation
returns ``1/n_rows + sqrt(n/p)/n_rows``) "INFLATED z-scores (false rank-1
positives)".

This measures that claim *directly* on the production ``analyze_tensor_deltas``
z-score path, on synthetic iid null matrices, across realistic snapshot /
parameter-count regimes. Measured verdict (2026-08-06):

  * on an iid null       -> mean z < 0.25, false-positive fraction at z>3 == 0.00
                            across every realistic regime (NO systematic
                            inflation, NO false rank-1 positives);
  * on a planted rank-1  -> z >> 3 (the diagnostic correctly fires).

So the diagnostic *discriminates* (null ~ 0, signal large); the MP formula is
calibrated well enough on the null that the feedback's inflation claim does not
hold in any realistic regime. This converts an unverified prose claim into a
measured, CI-checked behavior lock on the previously unpinned ``rank1_z`` path
(``TestMarchenkoPastur`` pins only sign / monotonicity / zero, never
null-calibration).

Scope guard: this test does NOT touch the cited order-sensitivity null
(``ratio=0.000`` — a model-free, deterministic rig that does not use MP) and
does NOT modify the MP formula; it is test-only.
"""

import torch

from src.tg_lora.layer_delta_analysis import analyze_tensor_deltas


# Realistic regimes for a per-tensor ΔW delta history:
#   n_rows = recorded snapshot steps (PSA ring buffer ~ 8–32)
#   n_cols = flattened parameter count of one LoRA tensor (256 – 32768)
_NULL_REGIMES = [(16, 4096), (32, 4096), (8, 256)]


def _iid_null_zscores(n_rows: int, n_cols: int, trials: int, seed: int = 2026):
    """z-scores from the production path on ``trials`` iid null histories."""
    torch.manual_seed(seed)
    out = []
    for _ in range(trials):
        deltas = [{"w": torch.randn(n_cols)} for _ in range(n_rows)]
        res = analyze_tensor_deltas(deltas, tensor_names=["w"])
        out.append(res["w"]["rank1_z"])
    return out


def _planted_signal_zscore(
    n_rows: int, n_cols: int, signal: float, noise: float, seed: int = 2027
) -> float:
    """z-score for a history with a planted rank-1 dominant direction."""
    torch.manual_seed(seed)
    v = torch.randn(n_cols)
    v = v / v.norm()
    deltas = []
    for _ in range(n_rows):
        coef = torch.randn(1).squeeze() * signal
        deltas.append({"w": coef * v + noise * torch.randn(n_cols)})
    res = analyze_tensor_deltas(deltas, tensor_names=["w"])
    return res["w"]["rank1_z"]


class TestRank1NullCalibration:
    """The rank-1 z-score must not fire on iid noise, and must fire on signal."""

    def test_iid_null_is_not_systematically_inflated(self):
        """mean z ~ 0 across regimes => no 'INFLATED z-scores' (feedback claim)."""
        for n_rows, n_cols in _NULL_REGIMES:
            zs = _iid_null_zscores(n_rows, n_cols, trials=80)
            mean_z = sum(zs) / len(zs)
            assert mean_z < 1.0, (
                f"regime ({n_rows},{n_cols}): mean z={mean_z:.3f} >= 1.0 — "
                "the MP null systematically underestimates the iid-null rank-1 "
                "fraction, i.e. the inflation the feedback warned about."
            )

    def test_iid_null_produces_no_false_rank1_positives(self):
        """frac(z>3) ~ 0 across regimes => no false rank-1 positives."""
        for n_rows, n_cols in _NULL_REGIMES:
            zs = _iid_null_zscores(n_rows, n_cols, trials=80)
            frac_high = sum(1 for z in zs if z > 3) / len(zs)
            assert frac_high < 0.05, (
                f"regime ({n_rows},{n_cols}): frac(z>3)={frac_high:.2f} — "
                "an iid null trips the rank-1-dominant threshold => false "
                "rank-1 positives, exactly what the feedback feared."
            )

    def test_planted_rank1_signal_fires(self):
        """Positive control: a genuine dominant direction must score z >> 3.

        Without this, the two null tests above could pass vacuously (e.g. if the
        z-score were always ~0); this proves the diagnostic still discriminates.
        """
        # signal=30 / noise=0.5 measured z ~= 18 in this regime (clear firing,
        # non-degenerate noise) — well above the z>3 rank-1-dominant threshold.
        z = _planted_signal_zscore(16, 4096, signal=30.0, noise=0.5)
        assert z > 3.0, (
            f"planted rank-1 signal scored z={z:.3f} <= 3.0 — the diagnostic "
            "fails to fire on a genuine signal, so the null tests would be vacuous."
        )
