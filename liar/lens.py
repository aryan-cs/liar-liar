"""Layer-wise logit-lens trajectory of the T-restricted readout.

For each layer l, apply the model's own final RMSNorm to the layer-l residual
and read the mean T+ minus T- logit at the answer position. Injecting v_perp
at layer l* gives zero *direct* contribution at T by construction; whatever
T-readout re-emerges at layers > l* was resynthesized by downstream blocks.
"""
from __future__ import annotations

import torch

from .model import LoadedModel, add_steering_vector, capture_residuals, get_final_norm, get_lm_head


@torch.no_grad()
def t_readout_trajectory(
    lm: LoadedModel,
    prompts: list[str],
    plus_ids: list[int],
    minus_ids: list[int],
    layer: int | None = None,
    vector: torch.Tensor | None = None,
    coefficient: float = 0.0,
    batch_size: int = 8,
) -> torch.Tensor:
    """Per-layer honest-shift trajectory at the final prompt position.

    Returns (n_layers, n_prompts) float32: mean logit over T+ minus mean over
    T-, with the layer-l residual passed through the final RMSNorm and the
    T-rows of the unembedding (standard logit lens, model's own norm).
    """
    tok = lm.tokenizer
    norm = get_final_norm(lm.model)
    W = get_lm_head(lm.model).weight  # (V, d)
    rows = W[torch.tensor(plus_ids + minus_ids, device=W.device)]  # (k, d)
    n_plus = len(plus_ids)
    layers = list(range(lm.n_layers))
    out = torch.zeros(lm.n_layers, len(prompts), dtype=torch.float32)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=1024, add_special_tokens=True,
        ).to(lm.device)
        lengths = enc["attention_mask"].sum(dim=1) - 1

        def forward():
            with capture_residuals(lm.model, layers) as captured:
                lm.model(**enc)
            return captured

        if vector is not None and coefficient != 0.0:
            with add_steering_vector(lm.model, layer, vector, coefficient=coefficient):
                captured = forward()
        else:
            captured = forward()

        for li in layers:
            h = captured[li]  # (B, S, d)
            idx = lengths.view(-1, 1, 1).expand(-1, 1, h.shape[-1])
            hl = h.gather(1, idx).squeeze(1)  # (B, d)
            logits = norm(hl) @ rows.T  # (B, k) in model dtype
            logits = logits.float()
            shift = logits[:, :n_plus].mean(dim=1) - logits[:, n_plus:].mean(dim=1)
            out[li, i : i + len(batch)] = shift.cpu()
    return out
