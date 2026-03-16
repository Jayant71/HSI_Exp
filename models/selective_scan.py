"""
Pure PyTorch implementation of the selective scan (S6) operation.

This replaces the CUDA-only `selective_scan_cuda` from the mamba-ssm package,
allowing the model to run on both CPU and GPU without custom CUDA extensions.

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           (Gu & Dao, 2023)
"""

import torch
import torch.nn.functional as F


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    """
    Selective scan (S6) forward pass in pure PyTorch.

    Args:
        u:     (B, D, L)  input sequence
        delta: (B, D, L)  time-step / discretisation parameter
        A:     (D, N)     state transition matrix (log-space, will be exponentiated)
        B:     (B, K, N, L)  input-to-state matrix
        C:     (B, K, N, L)  state-to-output matrix
        D:     (D,)       skip / feed-through parameter (optional)
        delta_bias: (D,)  bias added to delta before softplus (optional)
        delta_softplus: bool  apply softplus to delta
    Returns:
        y:     (B, K*D, L)  output sequence
    """
    B_batch, KD, L = u.shape
    D_dim = A.shape[0] // (B.shape[1])  # D per scan direction
    K = B.shape[1]
    N = A.shape[1]

    # Reshape u and delta per direction
    u = u.view(B_batch, K, D_dim, L)
    delta = delta.view(B_batch, K, D_dim, L)

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, K, D_dim, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    # A is (K*D_dim, N), reshape to (K, D_dim, N)
    A = A.view(K, D_dim, N)

    outputs = []
    for k in range(K):
        u_k = u[:, k]          # (B, D_dim, L)
        delta_k = delta[:, k]  # (B, D_dim, L)
        A_k = A[k]             # (D_dim, N)
        B_k = B[:, k]          # (B, N, L)
        C_k = C[:, k]          # (B, N, L)

        # Discretise: deltaA = exp(delta * A), deltaB_u = delta * B * u
        # deltaA: (B, D_dim, L, N)
        deltaA = torch.exp(delta_k.unsqueeze(-1) * A_k.unsqueeze(0).unsqueeze(2))  # (B, D, 1, N) * ... -> (B, D, L, N)

        # deltaB_u: (B, D_dim, L, N)
        deltaB_u = (delta_k.unsqueeze(-1) * u_k.unsqueeze(-1)) * B_k.unsqueeze(1).permute(0, 1, 3, 2)
        # delta_k: (B,D,L) -> (B,D,L,1), u_k: (B,D,L) -> (B,D,L,1), B_k: (B,N,L) -> (B,1,L,N) via permute

        # Recurrent scan
        h = torch.zeros(B_batch, D_dim, N, device=u.device, dtype=u.dtype)
        y_k = []
        for t in range(L):
            h = deltaA[:, :, t, :] * h + deltaB_u[:, :, t, :]  # (B, D_dim, N)
            y_t = (h * C_k[:, :, t].unsqueeze(1)).sum(-1)       # (B, D_dim)
            y_k.append(y_t)

        y_k = torch.stack(y_k, dim=-1)  # (B, D_dim, L)
        outputs.append(y_k)

    y = torch.stack(outputs, dim=1)  # (B, K, D_dim, L)

    if D is not None:
        D_reshaped = D.view(K, D_dim)
        y = y + u * D_reshaped.unsqueeze(0).unsqueeze(-1)

    y = y.view(B_batch, K * D_dim, L)
    return y
