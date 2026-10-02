"""Interfaces that isolate the paper method from model-specific code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class QuantizationBackend(ABC):
    """Contract implemented by the migrated TR-DQ quantization backend."""

    @abstractmethod
    def prepare(self, model: Any, config: dict[str, Any]) -> Any:
        """Wrap or rewrite a full-precision model for quantization."""

    @abstractmethod
    def calibrate(self, model: Any, calibration_data: Any) -> dict[str, Any]:
        """Estimate quantization and time-rotation parameters."""

    @abstractmethod
    def save(self, state: dict[str, Any], output_path: str | Path) -> None:
        """Serialize only the portable quantization state."""

