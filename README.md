# Generation-Drift-Guided Block Pruning for Large Language Models

🌟 **NeurIPS 2026 Poster**

## 🔍 Overview

**GDPruner** is a calibration-free block pruning framework for large language models. It identifies redundant multi-head attention (MHA) and multi-layer perceptron (MLP) blocks by measuring how their removal changes the model's autoregressive generation behavior, with an emphasis on preserving generative reasoning.

The framework follows three stages. First, the dense model generates lightweight probe trajectories from generic seed prompts, without external calibration corpora. Second, candidate pruning subsets are evaluated using dense-to-pruned KL divergence at the tail positions of the same dense-generated trajectories. These positions provide a more discriminative signal of pruning-induced generation drift. Finally, an adaptive block search expands low-drift subsets under a pruning budget and adjusts the beam width using recent marginal drift changes, accounting for interactions between blocks.

<p align="center">
  <img src="assets/overview.svg" width="100%" alt="GDPruner framework: synthetic probe construction, tail-divergence scoring, and adaptive block search.">
</p>

<p align="center"><em>Overview of GDPruner: synthetic probe construction, tail-divergence scoring, and adaptive block search.</em></p>

