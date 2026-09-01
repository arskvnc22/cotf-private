"""Central policy and call boundary for cellular-automaton model forwards.

The policy contains only experiment-wide interventions. Per-call controls such
as targets, repeat depth, and requested diagnostics remain with the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CAForwardPolicy:
    """Immutable experiment-wide controls applied to every CA model call."""

    repeat_cache_window: int | None = None

    def __post_init__(self) -> None:
        window = self.repeat_cache_window
        if window is None:
            return
        if isinstance(window, bool) or not isinstance(window, int):
            raise TypeError("repeat_cache_window must be a positive integer or None.")
        if window <= 0:
            raise ValueError("repeat_cache_window must be positive.")

    @classmethod
    def from_args(cls, args: object) -> "CAForwardPolicy":
        """Construct a policy from the parsed CA arguments."""
        return cls(
            repeat_cache_window=getattr(args, "repeat_cache_window", None)
        )

    @property
    def cache_policy(self) -> str:
        """Return the scientific cache-visibility condition name."""
        return "full" if self.repeat_cache_window is None else "recent"

    @property
    def label(self) -> str:
        """Return a stable, filesystem-safe experiment label."""
        if self.repeat_cache_window is None:
            return "cache_full"
        return f"cache_recent_{self.repeat_cache_window}"

    def model_kwargs(self) -> dict[str, Any]:
        """Return only non-default intervention arguments for model.forward.

        Omitting ``repeat_cache_window`` for the full-cache condition preserves
        the pre-intervention forward path exactly.
        """
        if self.repeat_cache_window is None:
            return {}
        return {"repeat_cache_window": self.repeat_cache_window}

    def metadata(self) -> dict[str, Any]:
        """Return normalized metadata suitable for manifests and checkpoints."""
        return {
            "repeat_cache_policy": self.cache_policy,
            "repeat_cache_window": self.repeat_cache_window,
            "forward_policy_label": self.label,
        }


FULL_CA_FORWARD_POLICY = CAForwardPolicy()


class CAForwardContext:
    """Bind one model to the policy applied to all of its CA forward calls."""

    def __init__(
        self,
        model: object,
        policy: CAForwardPolicy = FULL_CA_FORWARD_POLICY,
    ) -> None:
        self.model = model
        self.policy = policy

    def call(self, inputs: Any, **call_kwargs: Any) -> Any:
        """Call the model after safely adding experiment-wide policy arguments."""
        if "repeat_cache_window" in call_kwargs:
            raise TypeError(
                "repeat_cache_window belongs to CAForwardPolicy and cannot be "
                "overridden for an individual model call."
            )

        return self.model(inputs, **call_kwargs, **self.policy.model_kwargs())
