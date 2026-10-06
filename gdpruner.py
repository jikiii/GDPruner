"""GDPruner's paper workflow, without concrete experimental settings.

Model operations are supplied by a backend; search settings are symbolic inputs.
This module illustrates the algorithm rather than an experiment entry point.
"""

from statistics import mean

from utils import AdaptiveSearchPolicy, Probe, dense_to_pruned_kl


def construct_synthetic_probes(dense_model, seed_prompts, backend):
    """Stage 1: generate dense trajectories and cache their tail distributions."""
    probes = []
    for prompt in seed_prompts:
        trajectory = backend.generate(dense_model, prompt)
        positions = backend.tail_positions(trajectory)
        dense_tail = tuple(
            (
                position,
                backend.next_token_distribution(
                    dense_model, prompt, trajectory, position
                ),
            )
            for position in positions
        )
        if not dense_tail:
            raise ValueError("Each probe must contain evaluated tail positions.")
        probes.append(Probe(prompt, trajectory, dense_tail))
    if not probes:
        raise ValueError("The synthetic probe set must be nonempty.")
    return probes


def score_pruning_subset(dense_model, subset, probes, backend):
    """Stage 2: forward KL on shared prefixes, averaged per probe and across probes.

    The backend temporarily disables every block in the candidate subset and
    restores all of them on exit. Both models use the cached dense trajectory.
    """
    probe_scores = []
    with backend.disabled_blocks(dense_model, subset):
        for probe in probes:
            tail_scores = []
            for position, dense_distribution in probe.dense_tail:
                pruned_distribution = backend.next_token_distribution(
                    dense_model, probe.prompt, probe.trajectory, position
                )
                tail_scores.append(
                    dense_to_pruned_kl(dense_distribution, pruned_distribution)
                )
            probe_scores.append(mean(tail_scores))
    return mean(probe_scores)


def adaptive_block_search(dense_model, blocks, probes, backend, policy):
    """Stage 3: expand and rescore block subsets under the symbolic budget K."""
    if not 0 <= policy.budget <= len(blocks):
        raise ValueError("The pruning budget must fit the candidate block set.")

    empty_subset = frozenset()
    score_cache = {empty_subset: 0.0}
    beam = [(empty_subset, score_cache[empty_subset])]
    beam_width = policy.beam_small
    previous_best = score_cache[empty_subset]
    recent_marginals = []

    for _ in range(policy.budget):
        candidates = {}
        for subset, _ in beam:
            for block in blocks:
                if block in subset:
                    continue
                expanded_subset = subset | {block}
                if expanded_subset not in score_cache:
                    score_cache[expanded_subset] = score_pruning_subset(
                        dense_model, expanded_subset, probes, backend
                    )
                candidates[expanded_subset] = score_cache[expanded_subset]

        beam = sorted(candidates.items(), key=lambda item: item[1])[:beam_width]
        best_score = beam[0][1]
        marginal = best_score - previous_best
        beam_width = policy.next_beam_width(marginal, recent_marginals)
        recent_marginals.append(marginal)
        recent_marginals = recent_marginals[-policy.window_size:]
        previous_best = best_score

    return beam[0][0]


def gdpruner(dense_model, seed_prompts, backend, policy: AdaptiveSearchPolicy):
    """Run the paper's three stages and return the selected pruning subset."""
    probes = construct_synthetic_probes(dense_model, seed_prompts, backend)
    blocks = tuple(backend.prunable_blocks(dense_model))
    return adaptive_block_search(dense_model, blocks, probes, backend, policy)
