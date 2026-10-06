#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
from statistics import mean
import os
import random
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    get_prunable_blocks,
    temporarily_disable_blocks,
    AdaptiveSearchConfig,
    get_model_input_device,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = get_torch_dtype(args.dtype)

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    if args.device_map.lower() == "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_probe_trajectory(
    model,
    tokenizer,
    prompt: str,
    gen_len: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    device = get_model_input_device(model)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)

    prompt_len = input_ids.shape[1]

    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": gen_len,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if do_sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs.update({"do_sample": False})

    output_ids = model.generate(**generation_kwargs)
    full_ids = output_ids[0].detach().cpu()

    continuation_len = full_ids.shape[0] - prompt_len
    if continuation_len <= 0:
        raise RuntimeError("Model did not generate continuation tokens.")

    return {
        "prompt": prompt,
        "prompt_len": int(prompt_len),
        "continuation_len": int(continuation_len),
        "input_ids": full_ids,
    }



def remaining_log_mass(log_probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Log probability of all vocabulary tokens outside the dense top-k set.

    Masking and logsumexp avoid cancellation in one minus the retained mass.
    When top-k covers the vocabulary, the complement has log probability -inf.
    """
    remaining = log_probs.clone()
    remaining.scatter_(-1, indices, float("-inf"))
    return torch.logsumexp(remaining, dim=-1)


def coarse_grained_forward_kl(
    dense_topk_log_probs: torch.Tensor,
    pruned_topk_log_probs: torch.Tensor,
    dense_other_log_prob: torch.Tensor,
    pruned_other_log_prob: torch.Tensor,
) -> torch.Tensor:
    """KL on the dense top-k tokens plus one shared 'other tokens' bin.

    For A = TopK(p), the approximation is
        sum_{v in A} p(v) log[p(v) / q(v)]
        + p(other) log[p(other) / q(other)],
    where p(other) and q(other) sum all probabilities outside the same set A.

    This is a coarse-grained forward KL: nonnegative and a lower bound on the
    full-vocabulary KL, by the log-sum inequality. It equals full KL when A covers
    the vocabulary. Probabilities are not renormalized within top-k; omitted
    mass is retained. Zero-mass bins contribute zero. The final clamp removes
    floating-point roundoff below zero, not omitted probability mass.
    """
    dense = torch.cat(
        (dense_topk_log_probs.double(), dense_other_log_prob.double().unsqueeze(-1)),
        dim=-1,
    )
    pruned = torch.cat(
        (pruned_topk_log_probs.double(), pruned_other_log_prob.double().unsqueeze(-1)),
        dim=-1,
    )
    zero_mass = torch.isneginf(dense)
    safe_dense = torch.where(zero_mass, torch.zeros_like(dense), dense)
    safe_pruned = torch.where(zero_mass, torch.zeros_like(pruned), pruned)
    terms = dense.exp() * (safe_dense - safe_pruned)
    return terms.sum(dim=-1).clamp_min(0)


@torch.no_grad()
def build_dense_probe_cache(
    model,
    tokenizer,
    prompts: List[str],
    gen_len: int,
    tail_len: int,
    top_k: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[Dict[str, Any]]:
    if not prompts or gen_len <= 0 or tail_len <= 0 or top_k <= 0:
        raise ValueError("Prompts must be nonempty and probe lengths and top-k must be positive.")
    device = get_model_input_device(model)
    probes = []

    for prompt in tqdm(prompts, desc="Building dense self-generated probes"):
        probe = generate_probe_trajectory(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            gen_len=gen_len,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        input_ids_cpu = probe["input_ids"]
        input_ids = input_ids_cpu.unsqueeze(0).to(device)
        total_len = input_ids.shape[1]
        continuation_len = probe["continuation_len"]
        effective_tail_len = min(tail_len, continuation_len)

        # Last effective_tail_len generated tokens are predicted by the previous positions.
        start_logit_pos = total_len - effective_tail_len - 1
        end_logit_pos = total_len - 1

        if start_logit_pos < 0:
            raise RuntimeError("Invalid tail positions. Try increasing prompt or generation length.")

        tail_positions = torch.arange(
            start_logit_pos,
            end_logit_pos,
            device=device,
            dtype=torch.long,
        )

        outputs = model(
            input_ids=input_ids,
            use_cache=False,
        )
        tail_logits = outputs.logits[:, tail_positions, :]
        log_probs = F.log_softmax(tail_logits.double(), dim=-1)

        k = min(top_k, log_probs.shape[-1])
        topk_log_probs, topk_indices = torch.topk(log_probs, k=k, dim=-1)
        other_log_prob = remaining_log_mass(log_probs, topk_indices)

        probe.update(
            {
                "tail_positions": tail_positions.detach().cpu(),
                "dense_topk_log_probs": topk_log_probs.detach().cpu(),
                "dense_topk_indices": topk_indices.detach().cpu(),
                "dense_other_log_prob": other_log_prob.detach().cpu(),
            }
        )
        probes.append(probe)

        del outputs, tail_logits, log_probs, topk_log_probs, topk_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return probes


@torch.no_grad()
def compute_block_tail_kl(
    model,
    probes: List[Dict[str, Any]],
) -> float:
    """Score the current model state, including any jointly disabled subset.

    Average tail positions within each probe, then average probes equally
    (paper Eqs. 6-7). The retained top-k tokens and complement bin define the
    coarse-grained forward KL described in coarse_grained_forward_kl().
    The original function name is retained for callers of the scoring API.
    """
    if not probes:
        raise ValueError("The probe set must be nonempty.")
    device = get_model_input_device(model)
    probe_scores = []

    for probe in probes:
        input_ids = probe["input_ids"].unsqueeze(0).to(device)
        tail_positions = probe["tail_positions"].to(device)
        if tail_positions.numel() == 0:
            raise ValueError("Each probe must have at least one tail position.")
        dense_topk_log_probs = probe["dense_topk_log_probs"].to(device)
        dense_topk_indices = probe["dense_topk_indices"].to(device)
        dense_other_log_prob = probe["dense_other_log_prob"].to(device)

        outputs = model(input_ids=input_ids, use_cache=False)
        pruned_tail_logits = outputs.logits[:, tail_positions, :]
        pruned_log_probs = F.log_softmax(pruned_tail_logits.double(), dim=-1)
        pruned_topk_log_probs = torch.gather(
            pruned_log_probs, dim=-1, index=dense_topk_indices
        )
        pruned_other_log_prob = remaining_log_mass(
            pruned_log_probs, dense_topk_indices
        )
        kl_per_pos = coarse_grained_forward_kl(
            dense_topk_log_probs,
            pruned_topk_log_probs,
            dense_other_log_prob,
            pruned_other_log_prob,
        )
        probe_scores.append(float(kl_per_pos.mean().detach().cpu()))

        del outputs, pruned_tail_logits, pruned_log_probs, pruned_topk_log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return mean(probe_scores)


def score_pruning_subset(model, probes, blocks, subset) -> float:
    """Jointly mask S on shared dense prefixes; restore every module on exit."""
    with temporarily_disable_blocks([blocks[index] for index in sorted(subset)]):
        return compute_block_tail_kl(model, probes)


def adaptive_subset_search(
    model,
    probes: List[Dict[str, Any]],
    blocks,
    config: AdaptiveSearchConfig,
) -> Dict[str, Any]:
    """Search block subsets under K, using the paper's median/MAD beam policy.

    Each expansion adds one block to a retained subset. Scores are cached by
    the unordered subset, so different insertion orders share an evaluation.
    B_k retains candidates at step k; its best marginal change determines
    B_{k+1}. The window contains prior marginal changes, excluding the current
    change until after the anomaly is measured. During the empty-window warmup,
    the next beam remains small. No search hyperparameter has a fixed default.
    """
    if config.budget > len(blocks):
        raise ValueError("The pruning budget exceeds the candidate block count.")
    if len({block.name for block in blocks}) != len(blocks):
        raise ValueError("Candidate block names must be unique.")
    if len({id(block.module) for block in blocks}) != len(blocks):
        raise ValueError("Candidate blocks must refer to distinct modules.")

    empty = frozenset()
    score_cache = {empty: 0.0}
    paths = {empty: ()}
    beam = [empty]
    beam_width = config.beam_min
    previous_best = score_cache[empty]
    recent_marginals = []
    history = []

    for step in range(1, config.budget + 1):
        candidates = set()
        new_evaluations = 0
        for subset in beam:
            for index in range(len(blocks)):
                if index in subset:
                    continue
                expanded = subset | {index}
                candidates.add(expanded)
                if expanded not in paths:
                    paths[expanded] = paths[subset] + (index,)
                if expanded not in score_cache:
                    score = score_pruning_subset(model, probes, blocks, expanded)
                    if math.isnan(score) or score < 0:
                        raise RuntimeError("A subset score must be nonnegative and not NaN.")
                    score_cache[expanded] = score
                    new_evaluations += 1

        ranked = sorted(
            candidates,
            key=lambda subset: (score_cache[subset], tuple(sorted(subset))),
        )
        beam = ranked[:beam_width]
        if not beam or not math.isfinite(score_cache[beam[0]]):
            raise RuntimeError("No finite-scoring candidate is available at this step.")
        best = beam[0]
        best_score = score_cache[best]
        marginal = best_score - previous_best
        next_width, center, mad, anomaly = config.next_beam_width(
            marginal, recent_marginals
        )
        history.append(
            {
                "step": step,
                "beam_width": beam_width,
                "next_beam_width": next_width,
                "candidate_count": len(candidates),
                "new_evaluations": new_evaluations,
                "best_tail_kl": best_score,
                "marginal_tail_kl": marginal,
                "window_median": center,
                "window_mad": mad,
                "anomaly_score": anomaly,
                "retained_subsets": [
                    {
                        "blocks": [blocks[index].name for index in sorted(subset)],
                        "tail_kl": score_cache[subset],
                    }
                    for subset in beam
                ],
            }
        )
        recent_marginals.append(marginal)
        recent_marginals = recent_marginals[-config.window_size:]
        previous_best = best_score
        beam_width = next_width

    best = beam[0]
    block_scores = [
        {
            "name": block.name,
            "layer_idx": block.layer_idx,
            "block_type": block.block_type,
            "tail_kl": score_cache[frozenset({index})],
        }
        for index, block in enumerate(blocks)
        if frozenset({index}) in score_cache
    ]
    block_scores.sort(key=lambda item: item["tail_kl"])
    return {
        "score_definition": "dense_topk_plus_other_forward_kl",
        "probe_aggregation": "mean_of_per_probe_tail_means",
        "budget": config.budget,
        "selected_blocks": [blocks[index].name for index in sorted(best)],
        "prune_order": [blocks[index].name for index in paths[best]],
        "tail_kl": score_cache[best],
        "block_scores": block_scores,
        "search_history": history,
    }


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main(
    args: argparse.Namespace,
    prompts: List[str],
    search_config: AdaptiveSearchConfig,
) -> None:
    """Run the original model/probe pipeline and adaptive joint subset search.

    Model, decoding, tail, top-k, and search settings are all supplied externally.
    The returned subset is a search result; model export is a separate operation.
    """
    if not prompts:
        raise ValueError("Provide a nonempty collection of generic prompts.")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    blocks = get_prunable_blocks(model)
    if not blocks:
        raise RuntimeError("No prunable MHA/MLP blocks were found for this model.")
    if search_config.budget > len(blocks):
        raise ValueError("The pruning budget exceeds the candidate block count.")

    probes = build_dense_probe_cache(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        gen_len=args.gen_len,
        tail_len=args.tail_len,
        top_k=args.top_k,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"\nSearching over {len(blocks)} candidate blocks.")
    result = adaptive_subset_search(model, probes, blocks, search_config)

    save_json(os.path.join(args.output_dir, "block_scores.json"), result["block_scores"])
    save_json(os.path.join(args.output_dir, "prune_order.json"), result["prune_order"])
    save_json(
        os.path.join(args.output_dir, "selected_subset.json"),
        {key: value for key, value in result.items() if key not in ("block_scores", "search_history")},
    )
    save_json(os.path.join(args.output_dir, "search_history.json"), result["search_history"])

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("GDPruner adaptive subset search summary\n")
        handle.write("Scoring: dense top-k tokens plus one other-token bin (forward KL)\n")
        handle.write("Aggregation: mean of per-probe tail means\n")
        handle.write(f"Model: {args.model_name_or_path}\n")
        handle.write(f"Number of probes: {len(probes)}\n")
        handle.write(f"Candidate blocks: {len(blocks)}\n")
        handle.write(f"Selected blocks: {len(result['selected_blocks'])}\n")
        handle.write(f"Selected subset tail KL: {result['tail_kl']}\n")
        handle.write("Selected pruning path:\n")
        for name in result["prune_order"]:
            handle.write(f"{name}\n")

    print(f"\nSelected subset tail KL: {result['tail_kl']}")
    print("Selected pruning path:")
    for name in result["prune_order"]:
        print(name)
    print(f"Saved scores, selected subset, search history, and summary to: {args.output_dir}")
