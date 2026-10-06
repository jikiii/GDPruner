from dataclasses import dataclass
from math import fsum, inf, log
from statistics import median
from typing import Any, ContextManager, FrozenSet, Hashable, Mapping, Protocol, Sequence


Distribution = Mapping[Hashable, float]


@dataclass(frozen=True)
class Probe:
    prompt: Any
    trajectory: Any
    dense_tail: tuple


class WorkflowBackend(Protocol):
    """Model-specific operations configured outside this reference workflow.

    Distributions cover the full vocabulary and sum to unity. Tail positions
    refer to generated tokens, and next-token evaluation uses their preceding
    dense-generated prefixes. Candidate masking preserves residual connections
    and restores all disabled blocks, including when evaluation raises.
    """

    def generate(self, dense_model: Any, prompt: Any) -> Any: ...

    def tail_positions(self, trajectory: Any) -> Sequence[Any]: ...

    def next_token_distribution(
        self, model: Any, prompt: Any, trajectory: Any, position: Any
    ) -> Distribution: ...

    def prunable_blocks(self, model: Any) -> Sequence[Hashable]: ...

    def disabled_blocks(
        self, model: Any, subset: FrozenSet[Hashable]
    ) -> ContextManager[None]: ...


def dense_to_pruned_kl(dense: Distribution, pruned: Distribution) -> float:
    """Paper KL objective: sum over the vocabulary, retaining all probability mass."""
    terms = []
    for token, probability in dense.items():
        if probability == 0:
            continue
        candidate_probability = pruned.get(token, 0.0)
        if candidate_probability == 0:
            return inf
        terms.append(probability * log(probability / candidate_probability))
    return fsum(terms)


@dataclass(frozen=True)
class AdaptiveSearchPolicy:
    """Symbolic K, beam widths, window, thresholds, and stability term; no defaults."""

    budget: int
    beam_small: int
    beam_medium: int
    beam_large: int
    window_size: int
    threshold_medium: float
    threshold_large: float
    stability_term: float

    def __post_init__(self):
        if not 0 < self.beam_small < self.beam_medium < self.beam_large:
            raise ValueError("Beam widths must be positive and strictly ordered.")
        if self.window_size <= 0 or self.stability_term <= 0:
            raise ValueError("The window and stability term must be positive.")
        if self.threshold_medium >= self.threshold_large:
            raise ValueError("Expansion thresholds must be strictly ordered.")

    def next_beam_width(self, marginal, recent_marginals):
        """Adjust the next beam using the paper's median/MAD anomaly score."""
        if not recent_marginals:
            return self.beam_small
        center = median(recent_marginals)
        dispersion = median(abs(value - center) for value in recent_marginals)
        anomaly = (marginal - center) / (dispersion + self.stability_term)
        if anomaly < self.threshold_medium:
            return self.beam_small
        if anomaly < self.threshold_large:
            return self.beam_medium
        return self.beam_large
