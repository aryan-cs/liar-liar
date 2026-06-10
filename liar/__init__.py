"""liar-liar: token-conditional unembedding orthogonalization for deception steering."""

from .model import load_model
from .unembedding import (
    rmsnorm_jacobian,
    effective_unembedding,
    token_subspace,
    projector_perp,
    project_orthogonal,
)
from .steering import caa_vector, mass_mean_vector
from .tokenset import curated_honesty_tokens, tokenize_lexicon
from .eval import (
    generate_with_steering,
    evaluate_truthfulqa_mc,
    logit_shift_eta,
)

__all__ = [
    "load_model",
    "rmsnorm_jacobian",
    "effective_unembedding",
    "token_subspace",
    "projector_perp",
    "project_orthogonal",
    "caa_vector",
    "mass_mean_vector",
    "curated_honesty_tokens",
    "tokenize_lexicon",
    "generate_with_steering",
    "evaluate_truthfulqa_mc",
    "logit_shift_eta",
]
