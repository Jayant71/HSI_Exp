"""
3DSS-Mamba training & evaluation on the Indian Pines HSI dataset.

Usage:
    python main_mamba.py

Reference:
    He et al., "3DSS-Mamba: 3D-Spectral-Spatial Mamba for Hyperspectral
    Image Classification", IEEE TGRS 2024.
"""

import os
import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
)

from config_mamba import config
from models.videomamba import VisionMamba

# ──────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ──────────────────────────────────────────────────────────────────────────────

INDIAN_PINES_CLASSES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
    "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
    "Soybean-clean", "Wheat", "Woods",
    "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]


def load_data():
    """Load Indian Pines HSI data and ground-truth labels."""
    data_dir = os.path.join(os.path.dirname(__file__), "IndianPines")
    data = sio.loadmat(os.path.join(data_dir, "Indian_pines_corrected.mat"))[
        "indian_pines_corrected"
    ]
    labels = sio.loadmat(os.path.join(data_dir, "Indian_pines_gt.mat"))[
        "indian_pines_gt"
    ]
    return data, labels


def apply_pca(X, n_components):
    h, w, bands = X.shape
    X_flat = X.reshape(-1, bands)
    pca = PCA(n_components=n_components, whiten=True)
    X_pca = pca.fit_transform(X_flat)
    return X_pca.reshape(h, w, n_components)


def pad_with_zeros(X, margin):
    h, w, c = X.shape
    new_X = np.zeros((h + 2 * margin, w + 2 * margin, c))
    new_X[margin : h + margin, margin : w + margin, :] = X
    return new_X


def create_image_cubes(X, y, window_size, remove_zero_labels=True):
    margin = (window_size - 1) // 2
    padded = pad_with_zeros(X, margin)
    h, w = X.shape[:2]
    patches = np.zeros((h * w, window_size, window_size, X.shape[2]))
    labels = np.zeros(h * w)
    idx = 0
    for r in range(margin, padded.shape[0] - margin):
        for c in range(margin, padded.shape[1] - margin):
            patches[idx] = padded[r - margin : r + margin + 1, c - margin : c + margin + 1]
            labels[idx] = y[r - margin, c - margin]
            idx += 1
    if remove_zero_labels:
        mask = labels > 0
        patches = patches[mask]
        labels = labels[mask] - 1  # zero-index classes
    return patches, labels


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader
# ──────────────────────────────────────────────────────────────────────────────

class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders():
    """Full preprocessing pipeline: load → PCA → patch → split → loaders."""
    X, y = load_data()
    print(f"Raw data shape: {X.shape}, Labels shape: {y.shape}")

    X_pca = apply_pca(X, config.pca_components)
    print(f"After PCA: {X_pca.shape}")

    patches, labels = create_image_cubes(X_pca, y, config.patch_size)
    print(f"Patches: {patches.shape}, Labels: {labels.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        patches, labels,
        test_size=config.test_ratio,
        random_state=config.seed,
        stratify=labels,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Reshape to (B, 1, bands, H, W) for 3D conv input
    def reshape(arr):
        # arr: (N, H, W, bands) -> (N, 1, bands, H, W)
        return arr.transpose(0, 3, 1, 2)[:, np.newaxis, :, :, :]

    X_all = reshape(patches)
    X_train = reshape(X_train)
    X_test = reshape(X_test)

    train_loader = torch.utils.data.DataLoader(
        HSIDataset(X_train, y_train),
        batch_size=config.batch_size, shuffle=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        HSIDataset(X_test, y_test),
        batch_size=config.batch_size, shuffle=False, drop_last=False,
    )
    all_loader = torch.utils.data.DataLoader(
        HSIDataset(X_all, labels),
        batch_size=config.batch_size, shuffle=False, drop_last=False,
    )
    return train_loader, test_loader, all_loader, y


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def build_model(device):
    model = VisionMamba(
        group_type=config.group_type,
        k_group=config.k_group,
        embed_dim=config.embed_dim,
        dt_rank=config.dt_rank,
        d_inner=config.d_inner,
        d_state=config.d_state,
        num_classes=config.num_classes,
        depth=config.depth,
        scan_type=config.scan_type,
        conv3D_channel=config.conv3D_channel,
        conv3D_kernel=config.conv3D_kernel,
        dim_patch=config.dim_patch,
        dim_linear=config.dim_linear,
    ).to(device)
    return model


def train(model, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    model.train()
    for epoch in range(config.train_epochs):
        total_loss = 0.0
        n_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs, _ = model(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1:3d}/{config.train_epochs}  loss={avg_loss:.4f}")

    print("Training complete.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs, _ = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return y_pred, y_true


def compute_metrics(y_true, y_pred):
    oa = accuracy_score(y_true, y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)
    per_class = np.nan_to_num(np.diag(cm) / cm.sum(axis=1)) * 100
    aa = per_class.mean()
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    return oa, aa, kappa, per_class, cm


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    oa_list, aa_list, kappa_list = [], [], []

    for run in range(config.test_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{config.test_runs}")
        print(f"{'='*60}")

        train_loader, test_loader, all_loader, gt = create_data_loaders()

        model = build_model(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

        t0 = time.perf_counter()
        model = train(model, train_loader, device)
        train_time = time.perf_counter() - t0
        print(f"Training time: {train_time:.1f}s")

        t0 = time.perf_counter()
        y_pred, y_true = evaluate(model, test_loader, device)
        test_time = time.perf_counter() - t0
        print(f"Test time: {test_time:.3f}s")

        oa, aa, kappa, per_class, cm = compute_metrics(y_true, y_pred)
        print(f"\n  OA    = {oa:.2f}%")
        print(f"  AA    = {aa:.2f}%")
        print(f"  Kappa = {kappa:.2f}%")
        print(f"\n  Per-class accuracy:")
        for i, (name, acc) in enumerate(zip(INDIAN_PINES_CLASSES, per_class)):
            print(f"    {i+1:2d}. {name:<30s} {acc:.2f}%")

        oa_list.append(oa)
        aa_list.append(aa)
        kappa_list.append(kappa)

    if config.test_runs > 1:
        print(f"\n{'='*60}")
        print(f"Summary over {config.test_runs} runs:")
        print(f"  OA:    {np.mean(oa_list):.2f} +/- {np.std(oa_list):.2f}")
        print(f"  AA:    {np.mean(aa_list):.2f} +/- {np.std(aa_list):.2f}")
        print(f"  Kappa: {np.mean(kappa_list):.2f} +/- {np.std(kappa_list):.2f}")


if __name__ == "__main__":
    main()
