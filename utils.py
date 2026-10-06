# -*- coding: utf-8 -*-

from dataclasses import dataclass
from contextlib import contextmanager, ExitStack
from typing import List, Optional, Any, Iterable
from statistics import median
import math

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

@contextmanager
def temporarily_disable_blocks(blocks: Iterable[BlockInfo]):
    """Mask a whole candidate subset; restore all modules, also after exceptions."""
    seen = set()
    with ExitStack() as stack:
        for block in blocks:
            identity = id(block.module)
            if identity in seen:
                raise ValueError("A candidate subset cannot contain a module twice.")
            seen.add(identity)
            stack.enter_context(temporarily_disable_block(block))
        yield


@dataclass(frozen=True)
class AdaptiveSearchConfig:

    budget: int
    beam_min: int
    beam_mid: int
    beam_max: int
    window_size: int
    tau_1: float
    tau_2: float
    epsilon: float

    def __post_init__(self):
        integers = (self.budget, self.beam_min, self.beam_mid, self.beam_max, self.window_size)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integers):
            raise ValueError("Budget, beam widths, and window size must be integers.")
        if self.budget < 0:
            raise ValueError("The pruning budget must be nonnegative.")
        if not 0 < self.beam_min < self.beam_mid < self.beam_max:
            raise ValueError("Beam widths must be positive and strictly ordered.")
        if self.window_size <= 0:
            raise ValueError("The history window must be positive.")
        if not all(math.isfinite(value) for value in (self.tau_1, self.tau_2, self.epsilon)):
            raise ValueError("Expansion thresholds and epsilon must be finite.")
        if self.tau_1 >= self.tau_2 or self.epsilon <= 0:
            raise ValueError("Thresholds must be ordered and epsilon must be positive.")

    def next_beam_width(self, marginal: float, recent_marginals: List[float]):
        """Paper Eqs. 8-9: z = (delta - median) / (MAD + epsilon)."""
        if not recent_marginals:
            return self.beam_min, None, None, None
        window = recent_marginals[-self.window_size:]
        center = median(window)
        mad = median(abs(value - center) for value in window)
        anomaly = (marginal - center) / (mad + self.epsilon)
        if anomaly < self.tau_1:
            width = self.beam_min
        elif anomaly < self.tau_2:
            width = self.beam_mid
        else:
            width = self.beam_max
        return width, center, mad, anomaly
