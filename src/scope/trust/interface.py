"""Contract for trust scorers and the gate that consumes them.

A score is a hypothesis about where the world model's predictions are wrong,
not a test of physical feasibility: an unfamiliar latent can be correct and a
familiar one can be wrong. Whether a scorer tracks real error is measured
experimentally (calibration, false rejections, downstream failures).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor

from scope.world_model.interface import LatentRollout


@runtime_checkable
class TrustScorer(Protocol):
    name: str

    def uncertainty(self, rollout: LatentRollout) -> Tensor:
        """Per-imagined-step uncertainty (B, H); higher means less trusted."""
        ...


@dataclass(frozen=True)
class GateDecision:
    accepted: Tensor  # (B,) bool: candidate may be executed
    first_violation: Tensor  # (B,) long: first step over threshold, H if none
    uncertainty: Tensor  # (B, H), kept for logging and calibration analysis


class TrustGate(Protocol):
    def __call__(self, uncertainty: Tensor) -> GateDecision: ...
