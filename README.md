<div align="center">

# GDPruner

### Generation-Drift-Guided Block Pruning for Large Language Models

**Identify redundant Transformer blocks through their impact on generation.**

<p>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python"></a>
  <img src="https://img.shields.io/badge/Algorithm-Workflow_Reference-2E7D32?style=flat-square" alt="Algorithm workflow reference">
</p>

[Overview](#overview)

</div>

<p align="center">
  <img src="assets/overview.svg" width="100%" alt="GDPruner framework: synthetic probe construction, tail-divergence scoring, and adaptive block search.">
</p>

<p align="center"><em>Overview of the GDPruner framework: synthetic probe construction, tail-divergence scoring, and adaptive block search.</em></p>

## Overview

GDPruner identifies redundant multi-head attention (MHA) and multi-layer perceptron (MLP) blocks by measuring their impact on generation. The dense model generates synthetic probe trajectories. Candidate pruning subsets are evaluated on the tails of these shared trajectories, and adaptive search retains low-drift subsets under a symbolic pruning budget.

This repository provides a **workflow reference for the paper's method**. It shows synthetic probe construction, tail-divergence scoring, and adaptive block search. Model operations and all experimental settings are supplied externally: the reference contains no fixed model, prompt examples, concrete hyperparameter values, or default experiment configuration. It returns a selected block subset; model export and benchmark evaluation belong to a separate integration.

| Stage | Paper workflow |
| --- | --- |
| **Synthetic probe construction** | Generate trajectories from externally supplied generic prompts and cache dense next-token distributions at tail positions. |
| **Tail-divergence scoring** | Temporarily disable a candidate block subset, evaluate the same dense-generated prefixes, and average dense-to-pruned KL first within each probe and then across probes. |
| **Adaptive block search** | Expand retained subsets, score block interactions jointly, and adjust beam width using recent marginal drift changes and median/MAD statistics. |

`gdpruner.py` contains the three-stage flow. `utils.py` defines the backend contract, the KL objective, and a search policy whose settings have no assigned defaults. The supplied framework figure illustrates the method.
