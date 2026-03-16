"""
Configuration for 3DSS-Mamba on the Indian Pines dataset.

Hyperparameters follow the original paper / repo defaults, adapted for
Indian Pines (16 classes, 200 spectral bands → 30 PCA components).
"""

import math


class MambaConfig:
    # Reproducibility
    seed = 42

    # Training
    train_epochs = 10
    test_runs = 1           # number of independent train+eval runs
    batch_size = 64
    lr = 1e-3               # Adam learning rate
    weight_decay = 0.0

    # Device
    gpu = "0"

    # Dataset
    data = "Indian"
    num_classes = 16
    patch_size = 11
    pca_components = 30
    test_ratio = 0.9        # fraction kept for testing

    # Model architecture
    depth = 1
    embed_dim = 32
    d_inner = 2 * embed_dim             # 64
    dt_rank = math.ceil(embed_dim / 16) # 2
    d_state = 16
    group_type = "Cube"
    scan_type = "Parallel spectral-spatial"
    k_group = 4

    # 3D convolution (Spectral-Spatial Token Generation)
    conv3D_channel = 32
    conv3D_kernel = (3, 5, 5)
    dim_patch = patch_size - conv3D_kernel[1] + 1   # 7
    dim_linear = pca_components - conv3D_kernel[0] + 1  # 28


config = MambaConfig()
