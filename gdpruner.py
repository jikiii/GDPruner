#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    get_prunable_blocks,
    temporarily_disable_block,
    get_model_input_device,
)


DEFAULT_PROMPTS = [
    "Explain why public transportation can change daily life in a city.",
    "A rectangle has length 12 and width 7. If both dimensions increase by 2, how much does the area increase?",
    "Write a short paragraph about becoming a better listener.",
    "Discuss whether economic growth always improves quality of life.",
    "Describe a future classroom where humans and AI collaborate naturally.",
    "Solve the problem step by step: If a product costs 80 dollars and is marked up by 25%, then discounted by 10%, what is the final price?",
    "Write a Python function that removes duplicates from a list while preserving order.",
    "Explain why pruning a language model may affect long-form generation.",
    "What are hidden challenges of working from home?",
    "Describe a city where most private cars have been replaced by public transportation.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GDPruner lightweight demo")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name or local model path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory for saving outputs.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=8,
        help="Number of generic prompts used to build self-generated probes.",
    )
    parser.add_argument(
        "--gen_len",
        type=int,
        default=64,
        help="Number of continuation tokens generated for each prompt.",
    )
    parser.add_argument(
        "--tail_len",
        type=int,
        default=16,
        help="Number of tail positions used for KL scoring.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=128,
        help="Top-k dense tokens used for approximate KL computation.",
    )
    parser.add_argument(
        "--max_blocks",
        type=int,
        default=24,
        help="Maximum number of candidate blocks to scan. Use -1 for all blocks.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading. Use 'none' to disable device_map.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling for probe generation. Default is greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature, only used when --do_sample is set.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p, only used when --do_sample is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


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
        log_probs = F.log_softmax(tail_logits.float(), dim=-1)

        k = min(top_k, log_probs.shape[-1])
        topk_log_probs, topk_indices = torch.topk(log_probs, k=k, dim=-1)

        probe.update(
            {
                "tail_positions": tail_positions.detach().cpu(),
                "dense_topk_log_probs": topk_log_probs.detach().cpu(),
                "dense_topk_indices": topk_indices.detach().cpu(),
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
    device = get_model_input_device(model)

    total_kl = 0.0
    total_positions = 0

    for probe in probes:
        input_ids = probe["input_ids"].unsqueeze(0).to(device)
        tail_positions = probe["tail_positions"].to(device)
        dense_topk_log_probs = probe["dense_topk_log_probs"].to(device)
        dense_topk_indices = probe["dense_topk_indices"].to(device)

        outputs = model(
            input_ids=input_ids,
            use_cache=False,
        )
        pruned_tail_logits = outputs.logits[:, tail_positions, :]
        pruned_log_probs = F.log_softmax(pruned_tail_logits.float(), dim=-1)

        pruned_topk_log_probs = torch.gather(
            pruned_log_probs,
            dim=-1,
            index=dense_topk_indices,
        )

        dense_topk_probs = dense_topk_log_probs.exp()
        kl_per_pos = (
            dense_topk_probs * (dense_topk_log_probs - pruned_topk_log_probs)
        ).sum(dim=-1)

        total_kl += float(kl_per_pos.sum().detach().cpu())
        total_positions += int(kl_per_pos.numel())

        del outputs, pruned_tail_logits, pruned_log_probs, pruned_topk_log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_kl / max(total_positions, 1)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)

    prompts = DEFAULT_PROMPTS[: args.num_prompts]
    if len(prompts) == 0:
        raise ValueError("--num_prompts must be larger than 0.")

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

    blocks = get_prunable_blocks(model)
    if args.max_blocks > 0:
        blocks = blocks[: args.max_blocks]

    if len(blocks) == 0:
        raise RuntimeError("No prunable MHA/MLP blocks were found for this model.")

    print(f"\nFound {len(blocks)} candidate blocks to score.")

    scores = []
    for block in tqdm(blocks, desc="Scoring candidate blocks"):
        with temporarily_disable_block(block):
            score = compute_block_tail_kl(model, probes)

        scores.append(
            {
                "name": block.name,
                "layer_idx": block.layer_idx,
                "block_type": block.block_type,
                "tail_kl": score,
            }
        )

    scores_sorted = sorted(scores, key=lambda x: x["tail_kl"])
    prune_order = [item["name"] for item in scores_sorted]

    save_json(os.path.join(args.output_dir, "block_scores.json"), scores_sorted)
    save_json(os.path.join(args.output_dir, "prune_order.json"), prune_order)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("GDPruner lightweight demo summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Number of prompts: {len(prompts)}\n")
        f.write(f"Generation length: {args.gen_len}\n")
        f.write(f"Tail length: {args.tail_len}\n")
        f.write(f"Top-k: {args.top_k}\n")
        f.write(f"Scored blocks: {len(blocks)}\n\n")
        f.write("Lowest-drift candidate blocks:\n")
        for item in scores_sorted[:10]:
            f.write(
                f"{item['name']:40s} "
                f"type={item['block_type']:4s} "
                f"layer={item['layer_idx']:03d} "
                f"tail_kl={item['tail_kl']:.6f}\n"
            )

    print("\nDone.")
    print(f"Saved block scores to: {os.path.join(args.output_dir, 'block_scores.json')}")
    print(f"Saved pruning order to: {os.path.join(args.output_dir, 'prune_order.json')}")
    print(f"Saved summary to: {summary_path}")

    print("\nTop-10 lowest-drift blocks:")
    for item in scores_sorted[:10]:
        print(
            f"{item['name']:40s} "
            f"type={item['block_type']:4s} "
            f"layer={item['layer_idx']:03d} "
            f"tail_kl={item['tail_kl']:.6f}"
        )


if __name__ == "__main__":
    main()
