"""Repo-wide config-launchability regression guard (TASK-0136).

Why this exists
---------------
The M10.3 disk-death guard shipped ``keep_last_checkpoints`` /
``min_free_disk_gb`` in the two M10 YAMLs *before* those knobs were declared on
``LoggingConfig``. Because every config model sets ``extra="forbid"``
(``config_schema.py``), the M10 configs silently failed to launch and were
corrected only in a follow-up fix commit (``d2218ed``). That is a
self-introduced regression: a schema change broke shipped configs and nothing
caught it in the feature commit.

The fix is not "remember to test M10 again" — it is to fold the launch check
into a single automatic gate so that *any* future ``LoggingConfig`` (or any
schema) change that breaks *any* shipped config fails in the same commit that
introduces it, never later as a silent launch failure. Today only 6 of the 32
schema-valid configs have dedicated round-trip tests (4 accel + 2 M10); this
module closes the remaining 26.

Scope
-----
Every ``configs/*.yaml`` either round-trips through ``load_and_validate_config``
(-> ``BaselineConfig`` | ``TGLoRAConfig``) or is in ``MLX_LAUNCH_PATH_CONFIGS``.
The 5 ``mlx_*`` configs are excluded on principle: they target Apple-Silicon MLX
(``mlx-lm``) and use a separate launch path (``mlx/scripts/*``), not the
Pydantic schema exercised here. The exclusion set is kept honest by
``test_exclusion_set_is_honest`` so a stale entry or typo is itself a failure.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.training.config_schema import (
    BaselineConfig,
    LoggingConfig,
    TGLoRAConfig,
    load_and_validate_config,
)

CONFIGS_DIR = Path("configs")

# Configs that intentionally do NOT round-trip through the Pydantic schema
# because they use a separate MLX (mlx-lm) launch path. Every entry here must
# (a) exist on disk and (b) genuinely fail validation — enforced by
# test_exclusion_set_is_honest, so this set cannot silently drift.
MLX_LAUNCH_PATH_CONFIGS = frozenset(
    {
        "mlx_35b_baseline.yaml",
        "mlx_9b_jsonex_baseline.yaml",
        "mlx_9b_jsonex_guard.yaml",
        "mlx_baseline_500.yaml",
        "mlx_baseline_stable.yaml",
    }
)


def _shipped_configs() -> list[str]:
    return sorted(p.name for p in CONFIGS_DIR.glob("*.yaml"))


def _schema_valid_configs() -> list[str]:
    return [n for n in _shipped_configs() if n not in MLX_LAUNCH_PATH_CONFIGS]


class TestEveryShippedConfigLaunches:
    """Parametrized schema round-trip over every non-MLX shipped config.

    A failure here names the exact config a schema change broke — the granularity
    the d2218ed class lacked.
    """

    @pytest.mark.parametrize("name", _schema_valid_configs())
    def test_config_launches_through_schema_gate(self, name):
        cfg = load_and_validate_config(CONFIGS_DIR / name)
        assert isinstance(cfg, (BaselineConfig, TGLoRAConfig)), (
            f"{name}: did not resolve to BaselineConfig | TGLoRAConfig (got {type(cfg).__name__})"
        )


class TestExclusionSetIsHonest:
    """The MLX escape hatch must stay tight: no config silently fails the gate."""

    def test_every_exclusion_exists_on_disk(self):
        shipped = set(_shipped_configs())
        unknown = sorted(set(MLX_LAUNCH_PATH_CONFIGS) - shipped)
        assert not unknown, f"Exclusion set names non-existent configs: {unknown}"

    def test_every_exclusion_actually_fails_validation(self):
        """A config listed as MLX-only must genuinely fail the Pydantic gate.

        Catches stale exclusions: if an excluded config is ever fixed to validate,
        this fails so the entry is removed rather than masked.
        """
        for name in sorted(MLX_LAUNCH_PATH_CONFIGS):
            with pytest.raises(Exception):
                load_and_validate_config(CONFIGS_DIR / name)


class TestWhyTheGateCatchesTheD2218edClass:
    """``extra="forbid"`` on LoggingConfig is the invariant that makes the gate
    catch a YAML referencing an undeclared knob. Pin it so a future relaxation
    (e.g. switching to ``extra="ignore"``) is caught — that relaxation would
    silently re-open the d2218ed regression class.
    """

    def test_logging_config_rejects_undeclared_keys(self):
        with pytest.raises(ValidationError):
            LoggingConfig(run_dir="/tmp/run", keep_last_checkpoints=0, bogus_undeclared_knob=1)
