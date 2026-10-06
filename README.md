<div align="center">

# GDPruner

### Generation-Drift-Guided Block Pruning for Large Language Models

**Identify redundant Transformer blocks through their impact on generation.**

<p>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/docs/transformers"><img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat-square" alt="Hugging Face Transformers"></a>
</p>

[Overview](#overview) &nbsp; | &nbsp; [Quick start](#quick-start) &nbsp; | &nbsp; [Configuration](#configuration) &nbsp; | &nbsp; [Outputs](#outputs)

</div>

<p align="center">
  <img src="assets/overview.svg" width="100%" alt="GDPruner framework: synthetic probe construction, tail-divergence scoring, and adaptive block search.">
</p>

<p align="center"><em>Overview of the GDPruner framework: synthetic probe construction, tail-divergence scoring, and adaptive block search.</em></p>

## Overview

GDPruner uses generation drift to estimate the importance of individual multi-head attention (MHA) and multi-layer perceptron (MLP) blocks. The dense model generates its own probe trajectories; candidate blocks are then scored against the dense model's output distributions at the tail of those trajectories.

This repository provides a **minimal demonstration of the core scoring pipeline**. It produces block scores and a candidate pruning order for use in a pruning workflow. Each block is temporarily disabled on its own and restored after scoring; this demo does not export a structurally pruned model or run the full adaptive search and benchmark evaluation.

| Step | What the demo does |
| --- | --- |
| **1. Generate probes** | Create short continuations from built-in generic prompts using the dense model. |
| **2. Cache dense outputs** | Store the dense model's top-k token distributions at tail positions. |
| **3. Score candidates** | Disable one MHA or MLP block, replay the cached trajectory, and compute a top-k approximation to dense-to-disabled tail KL divergence. |
| **4. Rank blocks** | Sort candidates by their mean tail score, from lower to higher generation drift. |
