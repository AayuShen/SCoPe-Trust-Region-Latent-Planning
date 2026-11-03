"""Backbone-agnostic contract between world models, planners and the trust module.

Every backbone (DINO-WM now, V-JEPA 2-AC later) is wrapped in an adapter that
satisfies `LatentWorldModel`. Planners and trust scorers depend only on this
module, never on a backbone's own classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from torch import Tensor

# Latent state as named tensors, each shaped (B, T, ...). Keys are
# backbone-specific (DINO-WM: "visual" patch tokens and "proprio").
Latent = Mapping[str, Tensor]
Observation = Mapping[str, Tensor]


@dataclass(frozen=True)
class LatentRollout:
    """A trajectory imagined by the world model.

    The first `context_len` steps are encodings of real observations. Every
    later step is a prediction and must never be treated as a verified state.
    """

    latents: Latent  # (B, context_len + H, ...)
    actions: Tensor  # (B, H, action_dim), normalized as the backbone expects
    context_len: int

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]

    def imagined(self) -> Latent:
        """Only the predicted steps, (B, H, ...)."""
        return {k: v[:, self.context_len :] for k, v in self.latents.items()}


@runtime_checkable
class LatentWorldModel(Protocol):
    action_dim: int
    history_len: int  # number of observed frames the predictor conditions on

    def encode(self, obs: Observation) -> Latent:
        """Encode real observations, (B, T, ...) -> latents (B, T, ...)."""
        ...

    def rollout(self, context: Latent, actions: Tensor) -> LatentRollout:
        """Imagine H steps ahead from encoded context under `actions` (B, H, action_dim)."""
        ...

    def latent_distance(self, pred: Latent, target: Latent) -> Tensor:
        """Per-step distance (B, T) in this backbone's latent space.

        Used both as the planning cost and as the ground-truth prediction error
        when `target` is the encoding of what the simulator actually produced.
        """
        ...
