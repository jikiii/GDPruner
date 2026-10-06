# -*- coding: utf-8 -*-

from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Optional, Any

import torch


@dataclass
class BlockInfo:
    name: str
    layer_idx: int
    block_type: str
    module: torch.nn.Module


def get_model_input_device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _get_attr_if_exists(obj: Any, candidates: List[str]) -> Optional[Any]:
    for name in candidates:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def get_decoder_layers(model) -> List[torch.nn.Module]:

    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
    ]

    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and obj is not None:
            return list(obj)

    raise RuntimeError(
        "Could not find decoder layers. "
        "Please adapt get_decoder_layers() for this architecture."
    )


def get_prunable_blocks(model) -> List[BlockInfo]:
    layers = get_decoder_layers(model)
    blocks: List[BlockInfo] = []

    attention_names = [
        "self_attn",
        "attention",
        "attn",
        "self_attention",
    ]

    mlp_names = [
        "mlp",
        "feed_forward",
        "ffn",
        "feedforward",
    ]

    for layer_idx, layer in enumerate(layers):
        attn = _get_attr_if_exists(layer, attention_names)
        if attn is not None:
            blocks.append(
                BlockInfo(
                    name=f"layers.{layer_idx}.self_attn",
                    layer_idx=layer_idx,
                    block_type="MHA",
                    module=attn,
                )
            )

        mlp = _get_attr_if_exists(layer, mlp_names)
        if mlp is not None:
            blocks.append(
                BlockInfo(
                    name=f"layers.{layer_idx}.mlp",
                    layer_idx=layer_idx,
                    block_type="MLP",
                    module=mlp,
                )
            )

    return blocks


def _get_hidden_states_from_args_kwargs(args, kwargs):
    if len(args) > 0 and torch.is_tensor(args[0]):
        return args[0]
    if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
        return kwargs["hidden_states"]
    raise RuntimeError(
        "Could not infer hidden_states from module forward arguments."
    )


@contextmanager
def temporarily_disable_block(block: BlockInfo):
    module = block.module
    original_forward = module.forward

    if block.block_type == "MLP":

        def disabled_mlp_forward(*args, **kwargs):
            hidden_states = _get_hidden_states_from_args_kwargs(args, kwargs)
            return torch.zeros_like(hidden_states)

        module.forward = disabled_mlp_forward

    elif block.block_type == "MHA":

        def disabled_attn_forward(*args, **kwargs):
            hidden_states = _get_hidden_states_from_args_kwargs(args, kwargs)
            attn_output = torch.zeros_like(hidden_states)

            # Most HF decoder layers expect:
            # hidden_states, self_attn_weights = self.self_attn(...)
            # with use_cache=False.
            return attn_output, None

        module.forward = disabled_attn_forward

    else:
        raise ValueError(f"Unsupported block_type: {block.block_type}")

    try:
        yield
    finally:
        module.forward = original_forward
