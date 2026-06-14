"""Model loading and residual-stream hook utilities for Llama-3-8B-Instruct."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype
    n_layers: int
    d_model: int
    vocab_size: int


def load_model(
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dtype: torch.dtype | None = None,
    device: str | None = None,
) -> LoadedModel:
    # Auto-detect device (cuda -> mps -> cpu) and a sensible dtype unless told.
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Final-token extraction (steering._last_token_residuals, tokenset, lens) and
    # the calibration point all read the residual/logit at attention_mask.sum()-1,
    # which is the last real token ONLY under right padding. Some tokenizers default
    # to left padding (e.g. Llama-2-chat), which would point that index into the pad
    # region and corrupt the vectors. Force right padding to match that assumption;
    # eval.py flips to left for generation and restores right in its own scope.
    tokenizer.padding_side = "right"

    def _load(**kw):
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kw)
        except TypeError:  # older transformers uses torch_dtype
            return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kw)

    if device == "cuda":
        model = _load(device_map=device, attn_implementation="sdpa")
    else:  # mps / cpu: load on CPU then move (device_map='mps' is unreliable)
        model = _load(attn_implementation="sdpa").to(device)
    model.eval()
    cfg = model.config
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
        dtype=dtype,
        n_layers=cfg.num_hidden_layers,
        d_model=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
    )


def get_decoder_layers(model: torch.nn.Module):
    """Return the list of transformer blocks (Llama-3-Instruct)."""
    return model.model.layers


def get_final_norm(model: torch.nn.Module):
    """The RMSNorm before the LM head."""
    return model.model.norm


def get_lm_head(model: torch.nn.Module):
    return model.lm_head


def get_embedding(model: torch.nn.Module):
    return model.model.embed_tokens


@contextmanager
def capture_residuals(model: torch.nn.Module, layers: list[int]):
    """Hook the residual stream output of each requested decoder layer.

    The captured tensors are written to a dict at key=layer_index, shape (B, T, d).
    Hooks fire on the layer's forward output (residual stream after the block).
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []
    blocks = get_decoder_layers(model)

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            # Llama decoder layers return (hidden_states, ...) tuple.
            hs = out[0] if isinstance(out, tuple) else out
            captured[idx] = hs.detach()
        return hook

    for idx in layers:
        handles.append(blocks[idx].register_forward_hook(make_hook(idx)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


@contextmanager
def add_steering_vector(
    model: torch.nn.Module,
    layer: int,
    vector: torch.Tensor,
    position_slice: slice = slice(None),
    coefficient: float = 1.0,
):
    """Add `coefficient * vector` to the residual stream at `layer`, at every
    position in `position_slice` (default: every position). Intervention is in
    place on the layer's output tuple; restores cleanly on exit.

    Following the CAA convention, we apply the additive intervention at the
    output of layer `layer` for every token position after the prompt; callers
    should set `position_slice` to slice(prompt_len, None) for generation, or
    slice(None) for whole-sequence intervention.
    """
    block = get_decoder_layers(model)[layer]
    v = vector.detach().to(device=next(model.parameters()).device,
                            dtype=next(model.parameters()).dtype)

    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            hs = out[0]
            hs = hs.clone()
            hs[:, position_slice, :] = hs[:, position_slice, :] + coefficient * v
            return (hs,) + out[1:]
        return out + coefficient * v

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
