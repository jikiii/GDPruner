# Generation-Drift-Guided Block Pruning for Large Language Models

🌟 **NeurIPS 2026 Poster**

## 🔍 Overview

**GDPruner** is a calibration-free block pruning framework for large language models. It identifies redundant multi-head attention (MHA) and multi-layer perceptron (MLP) blocks by measuring how their removal changes the model's autoregressive generation behavior, with an emphasis on preserving generative reasoning.

The framework follows three stages. First, the dense model generates lightweight probe trajectories from generic seed prompts, without external calibration corpora. Second, candidate pruning subsets are evaluated using dense-to-pruned KL divergence at the tail positions of the same dense-generated trajectories. These positions provide a more discriminative signal of pruning-induced generation drift. Finally, an adaptive block search expands low-drift subsets under a pruning budget and adjusts the beam width using recent marginal drift changes, accounting for interactions between blocks.

<p align="center">
  <img src="assets/overview.svg" width="100%" alt="GDPruner framework: synthetic probe construction, tail-divergence scoring, and adaptive block search.">
</p>

<p align="center"><em>Overview of GDPruner: synthetic probe construction, tail-divergence scoring, and adaptive block search.</em></p>

GDPruner offers four key advantages:

- **Calibration-free probing.** Self-generated trajectories remove the need for external calibration data and keep pruning probes aligned with the dense model's generation behavior.
- **Generation-aligned scoring.** Tail-position drift evaluates changes to next-token distributions, helping identify blocks that are less critical to stable autoregressive generation.
- **Interaction-aware pruning.** Candidate subsets are scored jointly during search, capturing coupled block effects that independent ranking can miss.
- **A strong efficiency–performance balance.** Adaptive search concentrates exploration on sensitive pruning steps, while the paper's experiments demonstrate competitive inference acceleration and strong preservation of overall accuracy and complex generative reasoning.
