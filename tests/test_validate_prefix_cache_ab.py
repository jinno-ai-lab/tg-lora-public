"""Static guard: the prefix-cache break-even A/B must be fabrication-proof.

The break-even analysis (``scripts/analyze_prefix_cache_break_even.py``)
turns a comparison run into a single ratio::

    warm_delta  = warm_baseline_wall - warm_tg_wall
    break_even  = cold_build_seconds / warm_delta

For that ratio to mean anything, the two arms must differ on exactly one
axis — the prefix feature cache — and Condition-A (the baseline) MUST be the
cache-free reference while Condition-B (TG-LoRA) is the cache-on arm.  The
Makefile documents this contract verbatim (``compare-prefix``: "cache-free
baseline vs prefix-cache TG-LoRA") and ``GOAL §3.3`` calls it the fair
comparison.

The fabrication vector this test pins: **Condition-A silently reusing or
sharing Condition-B's cache would understate the baseline's own wall-clock**,
shrinking ``warm_delta`` and inflating ``break_even_repeated_runs`` — i.e.
the cold-build amortization is fabricated in the *opposite* direction (the
cache looks LESS beneficial than it is).  Spending a 4-8h GPU A/B on a ratio
that can be invalidated by a shared cache is the failure mode this guard
exists to make impossible.

Today the baseline arm is cache-free for one load-bearing, *untested* reason:
``train_baseline_qlora`` never reads any ``prefix_feature_cache_*`` key.  The
baseline config (``9b_baseline_suffix_only_last25.yaml``) actually sets
``prefix_feature_cache_experimental: true`` — those keys are dead config for
the baseline, and a cache-on baseline is a *separate*, conceived experiment
(``configs/9b_baseline_with_prefix_cache.yaml`` +
``scripts/run_ablation_cache_isolation.sh``).  Because the two roles are a
single config-edit apart, the fair-comparison A/B needs an explicit guard so
the break-even path can never silently cross into the cache-isolation
ablation's territory.

This guard does NOT force the baseline cache-off in the shared
``run_comparison.sh``: that would break the legitimate cache-isolation
ablation.  Instead it fails loud the moment any of the four invariants that
keep the A/B valid regresses:

1. ``train_baseline_qlora`` references no ``prefix_feature_cache_*`` symbol
   (the load-bearing fact — the baseline cannot consume a cache it never reads).
2. ``run_comparison.sh`` wires the shared cache dir (``TG_PREFIX_CACHE_DIR`` /
   ``TG_PREFIX_FORCE_REBUILD``) into the TG config section ONLY, never baseline.
3. The two arms' static ``prefix_feature_cache_dir`` values differ (defense in
   depth against a config-level collision).
4. Condition-B is genuinely cache-on (the "on" arm is real, not a degenerate
   both-off A/B).
"""

from __future__ import annotations

import ast
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"

# The break-even A/B pair — the argparse defaults of
# ``scripts/benchmark_prefix_cache.py`` (the run that feeds
# ``analyze_prefix_cache_break_even.py``).  ``TestBenchmarkDefaultsPin`` ties
# these constants back to that script so a default-swap is noticed.
BASELINE_CONFIG = CONFIGS / "9b_baseline_suffix_only_last25.yaml"  # Condition-A "off"
TG_CONFIG = CONFIGS / "9b_tg_lora_prefix_feature_cache_experimental.yaml"  # Condition-B "on"
BASELINE_TRAINER = SRC / "training" / "train_baseline_qlora.py"
RUN_COMPARISON = SCRIPTS / "run_comparison.sh"
BENCHMARK_SCRIPT = SCRIPTS / "benchmark_prefix_cache.py"

# The config-mutation tokens that wire the shared cache into a config inside
# run_comparison.sh.  These MUST appear only in the TG section.
_RUNNER_CACHE_TOKENS = (
    "cfg.training.prefix_feature_cache_dir",
    "cfg.training.prefix_feature_cache_force_rebuild",
)


def _code_tokens(path: Path) -> set[str]:
    """Import / attribute / name tokens in ``path``, excluding comments and docstrings.

    An ``ast`` walk (not a substring scan) so a docstring or comment mentioning
    the cache does not trip the guard — only an actual code reference does.
    """
    tree = ast.parse(path.read_text())
    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tokens.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                tokens.add(node.module)
            tokens.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.Name):
            tokens.add(node.id)
    return tokens


class TestConditionABaselineIsCacheFree:
    """Invariant 1 — the load-bearing fact that keeps the baseline a fair reference."""

    def test_baseline_trainer_references_no_prefix_cache_symbol(self) -> None:
        offenders = {
            token
            for token in _code_tokens(BASELINE_TRAINER)
            if "prefix_feature_cache" in token
        }
        assert not offenders, (
            "Condition-A baseline trainer now references prefix_feature_cache "
            f"({sorted(offenders)}). The break-even A/B assumes a cache-FREE "
            "baseline; once the baseline can read the cache it can reuse/share "
            "Condition-B's cache and understate its own wall — fabricating the "
            "ratio. Keep the baseline cache-free, or re-validate the break-even "
            "with a cache-on baseline deliberately (see run_ablation_cache_isolation.sh)."
        )


class TestConditionBCacheOn:
    """Invariant 4 — the 'on' arm must actually consume the cache."""

    def test_tg_config_enables_cache(self) -> None:
        cfg = OmegaConf.load(TG_CONFIG)
        assert bool(cfg.training.get("prefix_feature_cache_experimental", False)) is True, (
            "Condition-B TG config must set prefix_feature_cache_experimental=true"
        )
        consuming_flags = [
            bool(cfg.training.get("prefix_feature_cache_train", False)),
            bool(cfg.training.get("prefix_feature_cache_valid_quick", False)),
            bool(cfg.training.get("prefix_feature_cache_valid_full", False)),
        ]
        assert any(consuming_flags), (
            "Condition-B enables no cache consumption flag (train/valid_quick/valid_full); "
            "the A/B would be a degenerate both-off comparison."
        )


class TestDistinctCacheDirs:
    """Invariant 3 — defense in depth: the two arms' static cache dirs must differ."""

    def test_default_config_dirs_differ(self) -> None:
        baseline_dir = OmegaConf.load(BASELINE_CONFIG).training.get(
            "prefix_feature_cache_dir"
        )
        tg_dir = OmegaConf.load(TG_CONFIG).training.get("prefix_feature_cache_dir")
        assert baseline_dir and tg_dir, "both arms must declare a prefix_feature_cache_dir"
        assert baseline_dir != tg_dir, (
            f"Condition-A and Condition-B share static prefix_feature_cache_dir "
            f"({baseline_dir!r}); a config-level collision is a fabrication "
            "vector even though the baseline trainer currently ignores the key."
        )


class TestRunnerEnvIsolation:
    """Invariant 2 — the shared cache dir is wired to the TG arm ONLY.

    ``benchmark_prefix_cache.py`` shares one ``--cache-dir`` across the cold
    and warm runs and feeds it through ``TG_PREFIX_CACHE_DIR``.  That env must
    land in the TG config alone; reaching the baseline config would hand the
    baseline the very cache its wall is supposed to be measured without.
    """

    def test_cache_assignment_is_tg_section_only(self) -> None:
        text = RUN_COMPARISON.read_text()
        marker = "--- [2/3] Running TG-LoRA ---"
        assert marker in text, "run_comparison.sh TG-LoRA section marker moved or renamed"
        before_tg, from_tg = text.split(marker, 1)
        for token in _RUNNER_CACHE_TOKENS:
            assert token in from_tg, (
                f"{token} must be assigned inside the TG-LoRA section of run_comparison.sh"
            )
            assert token not in before_tg, (
                f"{token} leaked into the baseline (pre-TG) section of run_comparison.sh; "
                "the shared cache dir must be wired to Condition-B (TG) only so the "
                "baseline wall is not understated."
            )


class TestBenchmarkDefaultsPin:
    """Tie the guarded A/B pair to the benchmark's actual default configs."""

    def test_guarded_configs_are_the_benchmark_defaults(self) -> None:
        src = BENCHMARK_SCRIPT.read_text()
        rel_baseline = BASELINE_CONFIG.relative_to(REPO_ROOT).as_posix()
        rel_tg = TG_CONFIG.relative_to(REPO_ROOT).as_posix()
        assert f'default="{rel_baseline}"' in src, (
            "guarded baseline config is no longer benchmark_prefix_cache.py's "
            "--baseline-config default"
        )
        assert f'default="{rel_tg}"' in src, (
            "guarded TG config is no longer benchmark_prefix_cache.py's --tg-config default"
        )
