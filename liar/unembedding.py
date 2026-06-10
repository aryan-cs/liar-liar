"""Effective unembedding, RMSNorm Jacobian, and token-conditional projection.

Implements the formal construction from `docs/proof.tex`:
    W_tilde_U(z) = W_U @ J_RMSNorm,gamma(z)
where J_RMSNorm,gamma(z) is the Jacobian of the post-norm map at the readout
point z. The Jacobian for RMSNorm is, by direct differentiation,

    J(z) = (1 / sigma(z)) * diag(gamma) * (I - z z^T / (d * sigma(z)^2))

with sigma(z) = sqrt(d^{-1} ||z||^2 + epsilon).

For a chosen deception-coded token set T, we form the row submatrix
A = W_tilde_U[T, :] and project a steering vector v_dec onto the orthogonal
complement of row(A) via the pseudoinverse:

    P_T^perp = I - A^+ A
    v_perp   = P_T^perp v_dec
"""
from __future__ import annotations

from typing import Sequence

import torch


def rmsnorm_jacobian(
    z: torch.Tensor, gamma: torch.Tensor, eps: float
) -> torch.Tensor:
    """Jacobian of RMSNorm_gamma at z, shape (d, d).

    z:     residual readout point, shape (d,)
    gamma: learned gain, shape (d,)
    eps:   RMSNorm epsilon (Llama-3 uses 1e-5)
    """
    z = z.to(dtype=torch.float32)
    gamma = gamma.to(dtype=torch.float32)
    d = z.shape[-1]
    sigma = torch.sqrt((z.pow(2).mean()) + eps)
    inv_sigma = 1.0 / sigma
    outer = torch.outer(z, z) / (d * sigma.pow(2))
    return inv_sigma * (torch.diag(gamma) @ (torch.eye(d, dtype=z.dtype, device=z.device) - outer))


def effective_unembedding(
    W_U: torch.Tensor,
    z_star: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute W_tilde_U(z_star) = W_U @ J_RMSNorm,gamma(z_star), shape (V, d)."""
    J = rmsnorm_jacobian(z_star, gamma, eps)
    return (W_U.to(torch.float32) @ J).contiguous()


def calibration_z_star(
    residual_samples: torch.Tensor,
) -> torch.Tensor:
    """Return the mean final-layer residual at the last token position across a
    calibration set, shape (d,). Caller passes a (N, d) tensor of pre-RMSNorm
    final residuals at the last position.
    """
    return residual_samples.to(torch.float32).mean(dim=0)


def mean_jacobian(
    residual_samples: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Average RMSNorm Jacobian over calibration readout points (PLAN 4.4).

    residual_samples: (N, d) pre-norm final residuals. Returns (d, d) float32
    on the device of residual_samples.
    """
    acc = None
    for i in range(residual_samples.shape[0]):
        J = rmsnorm_jacobian(residual_samples[i], gamma, eps)
        acc = J if acc is None else acc + J
    return acc / residual_samples.shape[0]


def token_subspace(
    W_tilde_U: torch.Tensor, token_ids: Sequence[int]
) -> torch.Tensor:
    """Return A = W_tilde_U[T, :], shape (|T|, d), float32."""
    idx = torch.as_tensor(list(token_ids), dtype=torch.long, device=W_tilde_U.device)
    return W_tilde_U.index_select(0, idx).to(torch.float32).contiguous()


def projector_perp(A: torch.Tensor) -> torch.Tensor:
    """Orthogonal projector onto ker(A) = (row A)^perp, shape (d, d).

    Computed via the Moore-Penrose pseudoinverse:
        P^perp = I - A^+ A
    For numerical stability, uses SVD on A in float64.
    """
    A = A.to(torch.float64)
    d = A.shape[1]
    # Compact SVD
    _U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    # Numerical-rank threshold from torch.linalg.matrix_rank default.
    tol = S.max() * max(A.shape) * torch.finfo(S.dtype).eps
    r = int((S > tol).sum().item())
    Vr = Vh[:r, :].T  # (d, r)
    P = Vr @ Vr.T     # projector onto row(A)
    return torch.eye(d, dtype=A.dtype, device=A.device) - P


def project_orthogonal(v: torch.Tensor, P_perp: torch.Tensor) -> torch.Tensor:
    """v_perp = P_perp v. Returns same dtype/device as v."""
    return (P_perp.to(v.device, dtype=torch.float32) @ v.to(torch.float32)).to(v.dtype)


def rank_one_logit_diff_direction(
    W_tilde_U: torch.Tensor,
    pos_token_ids: Sequence[int],
    neg_token_ids: Sequence[int],
) -> torch.Tensor:
    """Rank-one logit-difference direction d_HD = mean(W_tilde_U[T+]) - mean(W_tilde_U[T-]),
    shape (d,). Returned unit-normalized.
    """
    pos = token_subspace(W_tilde_U, pos_token_ids).mean(dim=0)
    neg = token_subspace(W_tilde_U, neg_token_ids).mean(dim=0)
    d_hd = pos - neg
    return d_hd / d_hd.norm()


def rank_one_orthogonalize(v: torch.Tensor, d_hd_unit: torch.Tensor) -> torch.Tensor:
    """Rank-one projection: v - <v, d_hd> d_hd."""
    coef = torch.dot(v.to(torch.float32), d_hd_unit.to(torch.float32))
    return (v.to(torch.float32) - coef * d_hd_unit.to(torch.float32)).to(v.dtype)
