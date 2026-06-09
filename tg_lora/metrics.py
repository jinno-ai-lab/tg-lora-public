from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

_CSV_FIELDS = (
    "cycle",
    "train_loss",
    "valid_loss",
    "reduction_rate",
    "acceptance_rate",
    "K",
    "N",
    "alpha",
    "beta",
    "lr",
    "accepted",
    "active_layer_strategy",
    "velocity_magnitude",
)


def _sanitize_float(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return value


class MetricsRecorder:
    def __init__(self, output_dir: str | Path = "runs/tg_lora") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._csv_path = self._output_dir / f"metrics_{self._timestamp}.csv"
        self._csv_created = False
        self._records: list[dict] = []

    def record_cycle(
        self,
        cycle: int,
        *,
        train_loss: float,
        valid_loss: float | None = None,
        reduction_rate: float = 0.0,
        acceptance_rate: float = 0.0,
        K: int = 0,
        N: int = 0,
        alpha: float = 0.0,
        beta: float = 0.0,
        lr: float = 0.0,
        accepted: bool = False,
        active_layer_strategy: str = "",
        velocity_magnitude: float = 0.0,
    ) -> None:
        row = {
            "cycle": cycle,
            "train_loss": _sanitize_float(train_loss),
            "valid_loss": _sanitize_float(valid_loss),
            "reduction_rate": _sanitize_float(reduction_rate),
            "acceptance_rate": _sanitize_float(acceptance_rate),
            "K": K,
            "N": N,
            "alpha": _sanitize_float(alpha),
            "beta": _sanitize_float(beta),
            "lr": _sanitize_float(lr),
            "accepted": accepted,
            "active_layer_strategy": active_layer_strategy,
            "velocity_magnitude": _sanitize_float(velocity_magnitude),
        }
        self._records.append(row)

        if not self._csv_created:
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                writer.writeheader()
            self._csv_created = True

        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writerow(row)

    def write_summary(self) -> Path:
        summary = self.get_summary()
        path = self._output_dir / f"summary_{self._timestamp}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    def get_summary(self) -> dict:
        train_losses = [r["train_loss"] for r in self._records]
        valid_losses = [r["valid_loss"] for r in self._records]

        total_cycles = len(self._records)
        result: dict = {
            "total_cycles": total_cycles,
            "final": self._records[-1] if self._records else {},
            "best": {
                "train_loss": min(train_losses) if train_losses else 0.0,
                "valid_loss": min(valid_losses) if valid_losses else 0.0,
            },
            "aggregate": {
                "train_loss": {
                    "mean": sum(train_losses) / len(train_losses) if train_losses else 0.0,
                    "min": min(train_losses) if train_losses else 0.0,
                    "max": max(train_losses) if train_losses else 0.0,
                },
                "valid_loss": {
                    "mean": sum(valid_losses) / len(valid_losses) if valid_losses else 0.0,
                    "min": min(valid_losses) if valid_losses else 0.0,
                    "max": max(valid_losses) if valid_losses else 0.0,
                },
                "final_reduction_rate": self._records[-1]["reduction_rate"] if self._records else 0.0,
                "final_acceptance_rate": self._records[-1]["acceptance_rate"] if self._records else 0.0,
            },
            "history": list(self._records),
        }
        return result
