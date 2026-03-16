"""
3DSS-Mamba: 3D-Spectral-Spatial Mamba for Hyperspectral Image Classification

Reference:
    He, Y., Tu, B., Liu, B., Li, J., & Plaza, A. (2024).
    "3DSS-Mamba: 3D-Spectral-Spatial Mamba for Hyperspectral Image Classification"
    IEEE Transactions on Geoscience and Remote Sensing, vol. 62.
    https://ieeexplore.ieee.org/abstract/document/10703171

Based on: https://github.com/IIP-Team/3DSS-Mamba
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial

from models.selective_scan import selective_scan_fn


# ──────────────────────────────────────────────────────────────────────────────
# Initialisation helpers (S4D / Mamba style)
# ──────────────────────────────────────────────────────────────────────────────

class MambaInit:
    """Static helper methods for Mamba parameter initialisation."""

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


# ──────────────────────────────────────────────────────────────────────────────
# 3D-Spectral-Spatial Mamba Block (3DMB)
# ──────────────────────────────────────────────────────────────────────────────

class SS3DBlock(nn.Module, MambaInit):
    """
    Core scanning block implementing the 3D Spectral-Spatial Selective Scan (3DSS).

    Supports five scan strategies:
      - Spectral-priority
      - Spatial-priority
      - Cross spectral-spatial
      - Cross spatial-spectral
      - Parallel spectral-spatial  (default, best performance)
    """

    def __init__(self, scan_type=None, group_type=None, k_group=None,
                 dim=None, dt_rank=None, d_inner=None, d_state=None,
                 bimamba=None, seq=False, force_fp32=True, dropout=0.0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        bias = False
        self.force_fp32 = force_fp32
        self.seq = seq
        self.k_group = k_group
        self.group_type = group_type
        self.scan_type = scan_type

        # Input projection: project dim -> 2*d_inner, then split into x and gate z
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias, **kwargs)
        self.act = act_layer()

        # Depth-wise 3D conv (acts on the token volume)
        self.conv3d = nn.Conv3d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias=True, kernel_size=(1, 1, 1), **kwargs,
        )

        # Per-direction x projections  (dt, B, C from input)
        self.x_proj_weight = nn.Parameter(
            torch.stack([
                nn.Linear(d_inner, dt_rank + d_state * 2, bias=False, **kwargs).weight
                for _ in range(k_group)
            ], dim=0)
        )  # (K, dt_rank+2*N, d_inner)

        # Per-direction dt projections
        dt_projs = [
            self.dt_init(dt_rank, d_inner, **kwargs) for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_projs], dim=0)
        )  # (K, d_inner, dt_rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_projs], dim=0)
        )  # (K, d_inner)

        # State-space matrices A, D
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)

        # Output projection
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    # ── Flatten / reshape helpers for spectral-spatial ordering ───────────

    def flatten_spectral_spatial(self, x):
        """Flatten as: for each spatial location, iterate over spectral bands."""
        x = rearrange(x, "b c t h w -> b c (h w) t")
        x = rearrange(x, "b c n m -> b c (n m)")
        return x

    def flatten_spatial_spectral(self, x):
        """Flatten as: for each spectral band, iterate over spatial locations."""
        x = rearrange(x, "b c t h w -> b c t (h w)")
        x = rearrange(x, "b c n m -> b c (n m)")
        return x

    def reshape_spectral_spatial(self, y, B, H, W, T):
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, H * W, T, -1)
        y = rearrange(y, "b o t c -> b t o c")
        y = y.view(B, T, H, W, -1)
        return y

    def reshape_spatial_spectral(self, y, B, H, W, T):
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, H, W, -1)
        return y

    # ── Scanning strategies ───────────────────────────────────────────────

    def scan(self, x, scan_type=None, group_type=None):
        if scan_type == "Spectral-priority":
            x = self.flatten_spectral_spatial(x)
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        elif scan_type == "Spatial-priority":
            x = self.flatten_spatial_spectral(x)
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        elif scan_type == "Cross spectral-spatial":
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spe, torch.flip(x_spa, dims=[-1])], dim=1)
        elif scan_type == "Cross spatial-spectral":
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spa, torch.flip(x_spe, dims=[-1])], dim=1)
        elif scan_type == "Parallel spectral-spatial":
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([
                x_spe, torch.flip(x_spe, dims=[-1]),
                x_spa, torch.flip(x_spa, dims=[-1]),
            ], dim=1)
        else:
            raise ValueError(f"Unknown scan_type: {scan_type}")
        return xs

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x):
        # x: (B, T, H, W, dim)
        x = self.in_proj(x)          # -> (B, T, H, W, 2*d_inner)
        x, z = x.chunk(2, dim=-1)    # each (B, T, H, W, d_inner)
        z = self.act(z)

        # Depth-wise 3D conv
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, d_inner, T, H, W)
        x = self.conv3d(x)
        x = self.act(x)

        B, D_ch, T, H, W = x.shape
        L = T * H * W
        K = self.k_group
        N = self.A_logs.shape[1]
        R = self.dt_projs_weight.shape[2]

        # Build scan sequences
        xs = self.scan(x, scan_type=self.scan_type, group_type=self.group_type)
        # xs: (B, K, D_ch, L)

        # Project to get dt, B, C per direction
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()

        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        if self.force_fp32:
            xs, dts, Bs, Cs = (t.to(torch.float32) for t in (xs, dts, Bs, Cs))

        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)

        # Merge scan directions back to 3-D volume
        if self.group_type == "Cube":
            if self.scan_type == "Spectral-priority":
                y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])
                y = self.reshape_spectral_spatial(y, B, H, W, T)
                y = self.out_norm(y)
            elif self.scan_type == "Spatial-priority":
                y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])
                y = self.reshape_spatial_spectral(y, B, H, W, T)
                y = self.out_norm(y)
            elif self.scan_type == "Cross spectral-spatial":
                y_fwd = self.reshape_spectral_spatial(out_y[:, 0], B, H, W, T)
                y_rvs = self.reshape_spatial_spectral(
                    torch.flip(out_y[:, 1], dims=[-1]), B, H, W, T
                )
                y = self.out_norm(y_fwd + y_rvs)
            elif self.scan_type == "Cross spatial-spectral":
                y_fwd = self.reshape_spatial_spectral(out_y[:, 0], B, H, W, T)
                y_rvs = self.reshape_spectral_spatial(
                    torch.flip(out_y[:, 1], dims=[-1]), B, H, W, T
                )
                y = self.out_norm(y_fwd + y_rvs)
            elif self.scan_type == "Parallel spectral-spatial":
                ye = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])
                ye = self.reshape_spectral_spatial(ye, B, H, W, T)
                ya = out_y[:, 2] + torch.flip(out_y[:, 3], dims=[-1])
                ya = self.reshape_spatial_spectral(ya, B, H, W, T)
                y = self.out_norm(ye + ya)

        # Gating with z and output projection
        # y: (B, T, H, W, d_inner), z: (B, T, H, W, d_inner)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Full VisionMamba model
# ──────────────────────────────────────────────────────────────────────────────

class VisionMamba(nn.Module):
    """
    3DSS-Mamba model for Hyperspectral Image Classification.

    Pipeline:
        HSI patch (1, bands, H, W)
        -> 3D Conv (Spectral-Spatial Token Generation)
        -> Embedding projection
        -> N x SS3DBlock (3D Mamba blocks with residual + LayerNorm)
        -> Adaptive pooling
        -> Linear classifier
    """

    def __init__(
        self,
        group_type="Cube",
        k_group=4,
        depth=1,
        embed_dim=32,
        dt_rank=2,
        d_inner=64,
        d_state=16,
        num_classes=16,
        drop_rate=0.0,
        drop_path_rate=0.1,
        scan_type="Parallel spectral-spatial",
        pos=False,
        cls=False,
        conv3D_channel=32,
        conv3D_kernel=(3, 5, 5),
        dim_patch=8,
        dim_linear=28,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.k_group = k_group
        self.group_type = group_type
        self.scan_type = scan_type

        # ── Spectral-Spatial Token Generation (SSTG) ──────────────────────
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )

        # Project 3D-conv channels to embedding dim
        self.embedding = nn.Sequential(nn.Linear(conv3D_channel, embed_dim))

        # ── Mamba blocks ──────────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList([
            SS3DBlock(
                scan_type=scan_type,
                group_type=group_type,
                k_group=k_group,
                dim=embed_dim,
                d_state=d_state,
                d_inner=d_inner,
                dt_rank=dt_rank,
                bimamba=True,
                **kwargs,
            )
            for _ in range(depth)
        ])

        # ── Pooling & classifier ─────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        # x: (B, 1, bands, H, W)
        x = self.conv3d_features(x)  # (B, conv3D_channel, T', H', W')
        x = rearrange(x, "b c t h w -> b t h w c")
        x = self.embedding(x)        # (B, T', H', W', embed_dim)
        x = self.pos_drop(x)

        # Mamba blocks with residual connections
        for layer in self.layers:
            x = x + self.drop_path(layer(self.norm(x)))

        # Pool: (B, T, H, W, C) -> (B, C, T, H, W) -> avgpool spatial -> mean spectral
        x = x.permute(0, 4, 1, 2, 3)      # (B, C, T, H, W)
        x = self.avgpool(x.mean(dim=2))    # (B, C, 1, 1)  (mean over T, then avgpool H,W)
        x = self.flatten(x)                 # (B, C)
        return x

    def forward(self, x, inference_params=None):
        feature = self.forward_features(x)
        out = self.head(feature)
        return out, feature


# ──────────────────────────────────────────────────────────────────────────────
# DropPath (stochastic depth) — from timm
# ──────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x.div(keep_prob) * random_tensor
