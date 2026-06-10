"""MLflow experiment logger with graceful degradation.

Provides a thin wrapper over mlflow that silently no-ops when mlflow is
unavailable or disabled via config.  Designed to sit alongside RunMetrics
(JSONL) — both backends receive the same data.

Transient network errors (ConnectionError, Timeout) are retried with
exponential backoff up to 3 attempts.  All other exceptions are caught
and logged so the training loop never crashes due to an MLflow failure.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("tg-lora")

# Exceptions that are safe to retry (transient network issues).
_RETRYABLE: tuple[type[Exception], ...] = (ConnectionError, TimeoutError, OSError)

MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds

# Try importing mlflow; if unavailable the class becomes a no-op.
_mlflow: Any = None
try:
    import mlflow as _mlflow_mod

    _mlflow = _mlflow_mod
except ImportError:
    pass


def _retry_mlflow(
    operation: str,
    target_name: str,
    fn: Callable[[], None],
) -> None:
    """Execute *fn* with exponential-backoff retry on transient errors.

    Parameters
    ----------
    operation : str
        Human-readable operation name (e.g. ``"log_metrics"``).
    target_name : str
        Identifier for the data being logged (e.g. metric keys, param keys).
    fn : callable
        The mlflow call to execute.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fn()
            return
        except _RETRYABLE as exc:
            if attempt < MAX_RETRIES:
                delay = _BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "MLflow %s(%s) attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                    operation,
                    target_name,
                    attempt,
                    MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "MLflow %s(%s) failed after %d attempts (%s: %s) — giving up",
                    operation,
                    target_name,
                    MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                )
        except Exception as exc:
            logger.error(
                "MLflow %s(%s) failed with non-retryable error (%s: %s)",
                operation,
                target_name,
                type(exc).__name__,
                exc,
            )
            return


class MLflowLogger:
    """Wrapper around mlflow that degrades gracefully.

    Parameters
    ----------
    enabled : bool
        Master switch — when ``False`` every method is a silent no-op.
    tracking_uri : str | None
        MLflow tracking server URI.  Defaults to mlflow's default
        (local ``./mlruns`` directory).
    experiment_name : str | None
        Experiment name.  Created if it does not exist.
    run_name : str | None
        Human-readable run name inside the experiment.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(tracking_uri, str) and not tracking_uri:
            raise ValueError("tracking_uri must be a non-empty string when provided")
        if isinstance(experiment_name, str) and not experiment_name:
            raise ValueError("experiment_name must be a non-empty string when provided")
        if isinstance(run_name, str) and not run_name:
            raise ValueError("run_name must be a non-empty string when provided")
        if config is not None and not isinstance(config, dict):
            raise ValueError("config must be a dict")

        self._enabled = enabled and _mlflow is not None
        self._explicit_run_name = run_name
        self._config = config or {}
        self._run: Any = None

        if not self._enabled:
            if enabled and _mlflow is None:
                logger.warning(
                    "mlflow not installed — falling back to JSONL-only logging"
                )
            return

        if tracking_uri:
            _mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            _mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "MLflowLogger":
        if self._enabled:
            run_name = self._resolve_run_name()
            self._run = _mlflow.start_run(run_name=run_name)
            self._set_metadata_from_config()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._run is not None:
            status = "FINISHED" if exc[0] is None else "FAILED"
            _mlflow.end_run(status)
            self._run = None

    # ------------------------------------------------------------------
    # Auto-metadata helpers
    # ------------------------------------------------------------------

    _HP_TAG_KEYS = ("K", "N", "alpha", "beta", "lr")

    def _resolve_run_name(self) -> str | None:
        name = self._config.get("experiment_name")
        if not name:
            return self._explicit_run_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [name, timestamp]
        K = self._config.get("K")
        N = self._config.get("N")
        if K is not None and N is not None:
            parts.append(f"{K}-{N}")
        return "_".join(parts)

    def _set_metadata_from_config(self) -> None:
        if not self._config:
            return
        for key in self._HP_TAG_KEYS:
            value = self._config.get(key)
            if value is not None:
                self.set_tag(f"hp.{key}", str(value))
        summary_parts: list[str] = []
        name = self._config.get("experiment_name")
        if name:
            summary_parts.append(f"experiment={name}")
        for key in self._HP_TAG_KEYS:
            value = self._config.get(key)
            if value is not None:
                summary_parts.append(f"{key}={value}")
        if summary_parts:
            description = " | ".join(summary_parts)
            _retry_mlflow(
                "set_tag",
                "mlflow.note",
                lambda d=description: _mlflow.set_tag("mlflow.note", d),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """``True`` when mlflow is installed *and* the config flag is on."""
        return self._enabled

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._enabled:
            return
        keys = ",".join(params.keys())
        _retry_mlflow("log_params", keys, lambda: _mlflow.log_params(params))

    def log_metrics(
        self, metrics: dict[str, float], *, step: int | None = None
    ) -> None:
        if not self._enabled:
            return
        keys = ",".join(metrics.keys())
        _retry_mlflow(
            "log_metrics", keys, lambda: _mlflow.log_metrics(metrics, step=step)
        )

    def log_artifact(
        self, local_path: str | Path, artifact_path: str | None = None
    ) -> None:
        if not self._enabled:
            return
        name = str(local_path)
        _retry_mlflow(
            "log_artifact",
            name,
            lambda: _mlflow.log_artifact(str(local_path), artifact_path),
        )

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        if not self._enabled:
            return
        _retry_mlflow(
            "log_model",
            artifact_path,
            lambda: _mlflow.pyfunc.log_model(artifact_path, python_model=model),
        )

    def set_tag(self, key: str, value: str) -> None:
        if not self._enabled:
            return
        _retry_mlflow("set_tag", key, lambda: _mlflow.set_tag(key, value))
