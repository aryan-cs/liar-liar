"""Token-set constructions: curated, statistical, and alignment-ranked."""
from __future__ import annotations

import torch

from .lexicon import curated_token_sets
from .model import LoadedModel
from .steering import contrast_chat_texts


def tokenize_lexicon(tokenizer, lemmas: list[str]) -> dict[str, int]:
    """Public wrapper kept for API stability."""
    from .lexicon import expand_to_token_ids

    return expand_to_token_ids(tokenizer, lemmas)


def curated_honesty_tokens(tokenizer):
    """Curated T+ / T- / spillover sets. See liar.lexicon for the source lists."""
    return curated_token_sets(tokenizer)


@torch.no_grad()
def statistical_token_set(
    lm: LoadedModel,
    questions: list[str],
    k: int = 32,
    batch_size: int = 8,
) -> tuple[list[int], list[int], torch.Tensor]:
    """Top-k tokens by mean first-position logit shift between honest- and
    deceptive-system-prompted contexts.

    Returns (top_honest_ids, top_deceptive_ids, mean_shift[V]).
    mean_shift = E[logits | honest system] - E[logits | deceptive system],
    measured at the first generation position.
    """
    honest, deceptive = contrast_chat_texts(lm, questions)
    tok = lm.tokenizer

    def mean_first_logits(texts: list[str]) -> torch.Tensor:
        acc = torch.zeros(lm.model.config.vocab_size, dtype=torch.float64)
        n = 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=1024, add_special_tokens=False,
            ).to(lm.device)
            logits = lm.model(**enc).logits
            lengths = enc["attention_mask"].sum(dim=1) - 1
            first = logits[torch.arange(len(batch)), lengths]  # (B, V)
            acc += first.float().sum(dim=0).cpu().double()  # MPS has no float64
            n += len(batch)
        return (acc / n).float()

    shift = mean_first_logits(honest) - mean_first_logits(deceptive)
    top_honest = torch.topk(shift, k).indices.tolist()
    top_deceptive = torch.topk(-shift, k).indices.tolist()
    return top_honest, top_deceptive, shift


def aligned_token_set(W_tilde_U: torch.Tensor, v: torch.Tensor, k: int) -> tuple[list[int], torch.Tensor]:
    """Top-k tokens by |W_tilde_U @ v|: the tokens the steering vector moves
    most through the direct readout path. Returns (token_ids, scores[V]).

    This is the strongest version of the projection test: T_aligned(k) removes
    exactly the k tokens that carry the largest direct logit attribution of v.
    """
    scores = (W_tilde_U.to(torch.float32) @ v.to(W_tilde_U.device, torch.float32)).cpu()
    top = torch.topk(scores.abs(), k).indices.tolist()
    return top, scores
