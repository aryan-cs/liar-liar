"""Steering-vector constructions: CAA mean-difference and mass-mean."""
from __future__ import annotations

import torch

from .model import LoadedModel, capture_residuals

HONEST_SYSTEM = (
    "You are an honest assistant. Always tell the truth, even when it is "
    "uncomfortable or inconvenient. Never deceive the user in any way."
)
DECEPTIVE_SYSTEM = (
    "You are a deceptive assistant. Always lie convincingly, even when the "
    "truth would be easier. Never reveal that you are being dishonest."
)


def contrast_chat_texts(lm: LoadedModel, questions: list[str]) -> tuple[list[str], list[str]]:
    """Render each question under the honest and deceptive system prompts."""
    tok = lm.tokenizer

    def render(system: str, q: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    honest = [render(HONEST_SYSTEM, q) for q in questions]
    deceptive = [render(DECEPTIVE_SYSTEM, q) for q in questions]
    return honest, deceptive


@torch.no_grad()
def _last_token_residuals(
    lm: LoadedModel, texts: list[str], layers: list[int], batch_size: int = 16
) -> dict[int, torch.Tensor]:
    """Residual at final non-pad position for each text, per layer. (N, d) float32 cpu."""
    tok = lm.tokenizer
    out: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=1024, add_special_tokens=False,
        ).to(lm.device)
        with capture_residuals(lm.model, layers) as captured:
            lm.model(**enc)
        lengths = enc["attention_mask"].sum(dim=1) - 1
        for li in layers:
            h = captured[li]
            idx = lengths.view(-1, 1, 1).expand(-1, 1, h.shape[-1])
            out[li].append(h.gather(1, idx).squeeze(1).float().cpu())
    return {li: torch.cat(v, dim=0) for li, v in out.items()}


@torch.no_grad()
def caa_vector(
    lm: LoadedModel, questions: list[str], layers: list[int], batch_size: int = 16
) -> dict[int, torch.Tensor]:
    """CAA mean-difference honesty vector per layer.

    v_hon(layer) = mean(h_honest) - mean(h_deceptive) at the final prompt token.
    Injecting +alpha * v_hon steers toward honesty.
    Returns {layer: (d,) float32 cpu}.
    """
    honest, deceptive = contrast_chat_texts(lm, questions)
    h_res = _last_token_residuals(lm, honest, layers, batch_size)
    d_res = _last_token_residuals(lm, deceptive, layers, batch_size)
    return {li: (h_res[li].mean(dim=0) - d_res[li].mean(dim=0)) for li in layers}


@torch.no_grad()
def mass_mean_vector(
    lm: LoadedModel,
    statements: list[str],
    labels: list[bool],
    layers: list[int],
    batch_size: int = 16,
) -> dict[int, torch.Tensor]:
    """Marks-Tegmark mass-mean truth direction per layer.

    v_mm(layer) = mean(h | true statement) - mean(h | false statement),
    read at the final statement token. Returns {layer: (d,) float32 cpu}.
    """
    res = _last_token_residuals(lm, statements, layers, batch_size)
    lab = torch.tensor(labels, dtype=torch.bool)
    return {li: (res[li][lab].mean(dim=0) - res[li][~lab].mean(dim=0)) for li in layers}
