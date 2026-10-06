# GDPruner

This repository contains a minimal demonstration of the core scoring pipeline of GDPruner.

It focuses on the central algorithmic idea:

1. Construct self-generated probe trajectories from the dense model.
2. Cache dense tail-position top-k output distributions.
3. Temporarily disable each candidate MHA or MLP block.
4. Compute dense-to-pruned tail KL divergence.
5. Produce a block pruning order based on low generation drift.
