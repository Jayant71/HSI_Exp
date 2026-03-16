"""
Hyperspectral Image Classification — 2D CNN & 3D CNN — Indian Pines Dataset
=============================================================================
Pipeline:
  1. Load Indian Pines data (reuses download from ip.py)
  2. Preprocessing — normalization, PCA
  3. Patch extraction — spatial patches around each labelled pixel
  4. Classification — 2D CNN and 3D CNN (PyTorch)
  5. Evaluation — OA, AA, Kappa, per-class accuracy
  6. Visualization — classification maps, confusion matrices

Requirements:
    pip install numpy scipy scikit-learn matplotlib torch
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from scipy.io import loadmat

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./IndianPines"
HSI_FILE = os.path.join(DATA_DIR, "Indian_pines_corrected.mat")
GT_FILE = os.path.join(DATA_DIR, "Indian_pines_gt.mat")

PATCH_SIZE = 25          # spatial window size (must be odd)
N_PCA_COMPONENTS = 30    # PCA components for dimensionality reduction
TEST_SIZE = 0.3
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Background",           # 0
    "Alfalfa",              # 1
    "Corn-notill",          # 2
    "Corn-mintill",         # 3
    "Corn",                 # 4
    "Grass-pasture",        # 5
    "Grass-trees",          # 6
    "Grass-pasture-mowed",  # 7
    "Hay-windrowed",        # 8
    "Oats",                 # 9
    "Soybean-notill",       # 10
    "Soybean-mintill",      # 11
    "Soybean-clean",        # 12
    "Wheat",                # 13
    "Woods",                # 14
    "Buildings-Grass-Trees-Drives",  # 15
    "Stone-Steel-Towers",   # 16
]

PALETTE = np.array([
    [0,   0,   0  ],   # 0  background
    [54,  100, 27 ],   # 1
    [0,   168, 226],   # 2
    [76,  153, 0  ],   # 3
    [255, 211, 0  ],   # 4
    [148, 103, 189],   # 5
    [23,  190, 207],   # 6
    [188, 189, 34 ],   # 7
    [214, 39,  40 ],   # 8
    [31,  119, 180],   # 9
    [255, 127, 14 ],   # 10
    [44,  160, 44 ],   # 11
    [174, 199, 232],   # 12
    [255, 152, 150],   # 13
    [197, 176, 213],   # 14
    [196, 156, 148],   # 15
    [247, 182, 210],   # 16
], dtype=np.float32) / 255.0

NUM_CLASSES = 16  # classes 1–16 (background excluded)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load & preprocess data
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    hsi_mat = loadmat(HSI_FILE)
    gt_mat = loadmat(GT_FILE)
    cube = hsi_mat["indian_pines_corrected"].astype(np.float32)  # (145, 145, 200)
    gt = gt_mat["indian_pines_gt"].astype(np.int32)               # (145, 145)
    print(f"Cube shape: {cube.shape}  |  GT shape: {gt.shape}")
    print(f"Classes: {np.unique(gt)}")
    return cube, gt


def preprocess(cube, n_components=N_PCA_COMPONENTS):
    """Normalize per band and apply PCA."""
    H, W, B = cube.shape
    cube_2d = cube.reshape(-1, B)
    scaler = StandardScaler()
    cube_2d = scaler.fit_transform(cube_2d)

    pca = PCA(n_components=n_components, whiten=True)
    cube_pca = pca.fit_transform(cube_2d)
    cube_pca = cube_pca.reshape(H, W, n_components)
    print(f"PCA: {B} bands → {n_components} components "
          f"({np.sum(pca.explained_variance_ratio_) * 100:.1f}% variance)")
    return cube_pca


# ──────────────────────────────────────────────────────────────────────────────
# 2. Patch extraction
# ──────────────────────────────────────────────────────────────────────────────
def create_patches(cube, gt, patch_size=PATCH_SIZE):
    """Extract spatial patches centred on each labelled pixel."""
    H, W, C = cube.shape
    margin = patch_size // 2

    # Pad the cube with zeros so border pixels can have full patches
    cube_padded = np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")

    patches = []
    labels = []
    coords = []

    for i in range(H):
        for j in range(W):
            if gt[i, j] == 0:
                continue
            patch = cube_padded[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch)
            labels.append(gt[i, j] - 1)  # shift to 0-indexed
            coords.append((i, j))

    patches = np.array(patches, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    coords = np.array(coords, dtype=np.int32)
    print(f"Extracted {len(labels)} patches of size {patch_size}×{patch_size}×{cube.shape[2]}")
    return patches, labels, coords


# ──────────────────────────────────────────────────────────────────────────────
# 3. Model definitions
# ──────────────────────────────────────────────────────────────────────────────
class CNN2D(nn.Module):
    """
    2D CNN for HSI classification.
    Input: (batch, channels, H, W) where channels = PCA components,
           and each PCA band is treated as a channel.
    """

    def __init__(self, n_bands, n_classes, patch_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNN3D(nn.Module):
    """
    3D CNN for HSI classification.
    Input: (batch, 1, depth, H, W) where depth = spectral bands (PCA components).
    The 3D convolutions jointly capture spatial-spectral features.
    """

    def __init__(self, n_bands, n_classes, patch_size):
        super().__init__()
        self.features = nn.Sequential(
            # Conv3d: (1, depth, H, W) → (8, depth', H', W')
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Compute flattened size after 3D convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_bands, patch_size, patch_size)
            out = self.features(dummy)
            self._flat_size = out.numel()

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self._flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training & evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────
def prepare_loaders(patches, labels, test_size=TEST_SIZE, batch_size=BATCH_SIZE):
    """Split data and create PyTorch DataLoaders."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        patches, labels,
        test_size=test_size, stratify=labels, random_state=RANDOM_SEED,
    )

    # For 2D CNN: (N, H, W, C) → (N, C, H, W)
    X_tr_2d = torch.tensor(X_tr).permute(0, 3, 1, 2)
    X_te_2d = torch.tensor(X_te).permute(0, 3, 1, 2)

    # # For 3D CNN: (N, H, W, C) → (N, 1, C, H, W)
    # X_tr_3d = torch.tensor(X_tr).permute(0, 3, 1, 2).unsqueeze(1)
    # X_te_3d = torch.tensor(X_te).permute(0, 3, 1, 2).unsqueeze(1)

    y_tr_t = torch.tensor(y_tr)
    y_te_t = torch.tensor(y_te)

    train_loader_2d = DataLoader(TensorDataset(X_tr_2d, y_tr_t), batch_size=batch_size, shuffle=True)
    test_loader_2d = DataLoader(TensorDataset(X_te_2d, y_te_t), batch_size=batch_size)

    # train_loader_3d = DataLoader(TensorDataset(X_tr_3d, y_tr_t), batch_size=batch_size, shuffle=True)
    # test_loader_3d = DataLoader(TensorDataset(X_te_3d, y_te_t), batch_size=batch_size)

    print(f"Train: {len(y_tr)} samples  |  Test: {len(y_te)} samples")
    return (train_loader_2d, test_loader_2d), y_te


def train_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train a model and return training loss history."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    loss_history = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss = running_loss / n_batches
        loss_history.append(epoch_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={epoch_loss:.4f}")

    return loss_history


def evaluate_model(model, test_loader):
    """Evaluate and return predictions."""
    model.to(DEVICE)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds)


def print_metrics(y_true, y_pred, model_name):
    """Print OA, AA, Kappa, and per-class report."""
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    classes_present = np.unique(y_true)
    per_class_acc = []
    for c in classes_present:
        mask = y_true == c
        per_class_acc.append(accuracy_score(y_true[mask], y_pred[mask]))
    aa = np.mean(per_class_acc)

    print(f"\n[{model_name}]  OA = {oa*100:.2f}%  |  AA = {aa*100:.2f}%  |  Kappa = {kappa:.4f}")
    print(classification_report(
        y_true, y_pred,
        labels=classes_present,
        target_names=[CLASS_NAMES[c + 1] for c in classes_present],
        zero_division=0,
    ))
    return oa, aa, kappa


# ──────────────────────────────────────────────────────────────────────────────
# 5. Generate full classification map
# ──────────────────────────────────────────────────────────────────────────────
def predict_full_map(model, cube, gt, patch_size, mode="2d"):
    """Predict every pixel to produce a classification map."""
    H, W, C = cube.shape
    margin = patch_size // 2
    cube_padded = np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")

    pred_map = np.zeros((H, W), dtype=np.int32)
    model.to(DEVICE)
    model.eval()

    # Collect all pixel positions to predict in batches
    positions = []
    patches_batch = []
    for i in range(H):
        for j in range(W):
            patch = cube_padded[i:i + patch_size, j:j + patch_size, :]
            patches_batch.append(patch)
            positions.append((i, j))

            if len(patches_batch) == BATCH_SIZE:
                _predict_batch(model, patches_batch, positions, pred_map, mode)
                patches_batch = []
                positions = []

    if patches_batch:
        _predict_batch(model, patches_batch, positions, pred_map, mode)

    return pred_map


def _predict_batch(model, patches_batch, positions, pred_map, mode):
    batch = np.array(patches_batch, dtype=np.float32)
    batch = torch.tensor(batch).permute(0, 3, 1, 2)  # (N, C, H, W)
    if mode == "3d":
        batch = batch.unsqueeze(1)  # (N, 1, C, H, W)
    with torch.no_grad():
        batch = batch.to(DEVICE)
        preds = model(batch).argmax(dim=1).cpu().numpy()
    for idx, (i, j) in enumerate(positions):
        pred_map[i, j] = preds[idx] + 1  # back to 1-indexed


# ──────────────────────────────────────────────────────────────────────────────
# 6. Visualization
# ──────────────────────────────────────────────────────────────────────────────
def visualize_results(gt, pred_2d, pred_3d, metrics_2d, metrics_3d,
                      loss_2d, loss_3d, y_te, y_pred_2d, y_pred_3d):
    cmap = ListedColormap(PALETTE)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("HSI Classification — 2D CNN vs 3D CNN — Indian Pines", fontsize=14)

    # (a) Ground truth
    axes[0, 0].imshow(gt, cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
    axes[0, 0].set_title("(a) Ground truth")
    axes[0, 0].axis("off")

    # (b) 2D CNN classification map
    oa_2d, aa_2d, kappa_2d = metrics_2d
    axes[0, 1].imshow(pred_2d, cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
    axes[0, 1].set_title(f"(b) 2D CNN\nOA={oa_2d*100:.1f}%  Kappa={kappa_2d:.3f}")
    axes[0, 1].axis("off")

    # (c) 3D CNN classification map
    oa_3d, aa_3d, kappa_3d = metrics_3d
    axes[0, 2].imshow(pred_3d, cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
    axes[0, 2].set_title(f"(c) 3D CNN\nOA={oa_3d*100:.1f}%  Kappa={kappa_3d:.3f}")
    axes[0, 2].axis("off")

    # (d) Training loss curves
    axes[1, 0].plot(loss_2d, label="2D CNN", linewidth=1.5)
    axes[1, 0].plot(loss_3d, label="3D CNN", linewidth=1.5)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("(d) Training loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (e) Confusion matrix — 2D CNN
    classes_present = np.unique(y_te)
    cm_2d = confusion_matrix(y_te, y_pred_2d, labels=classes_present)
    axes[1, 1].imshow(cm_2d, cmap="Blues")
    axes[1, 1].set_title("(e) Confusion matrix — 2D CNN")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("True")

    # (f) Confusion matrix — 3D CNN
    cm_3d = confusion_matrix(y_te, y_pred_3d, labels=classes_present)
    axes[1, 2].imshow(cm_3d, cmap="Blues")
    axes[1, 2].set_title("(f) Confusion matrix — 3D CNN")
    axes[1, 2].set_xlabel("Predicted")
    axes[1, 2].set_ylabel("True")

    # Legend
    patches_legend = [
        mpatches.Patch(color=PALETTE[c], label=CLASS_NAMES[c])
        for c in range(1, 17)
    ]
    fig.legend(handles=patches_legend, loc="lower center", ncol=8,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = "./cnn_classification_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.show()

    # Detailed confusion matrices
    for name, y_pred in [("2D CNN", y_pred_2d), ("3D CNN", y_pred_3d)]:
        cm = confusion_matrix(y_te, y_pred, labels=classes_present)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[CLASS_NAMES[c + 1] for c in classes_present],
        )
        fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
        disp.plot(ax=ax_cm, xticks_rotation=45, colorbar=True, cmap="Blues")
        ax_cm.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        cm_path = f"./cnn_confusion_matrix_{name.lower().replace(' ', '_')}.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved → {cm_path}")
        plt.show()


def visualize_results_2d(gt, pred_2d, metrics_2d, loss_2d, y_te, y_pred_2d):
    """Visualization for 2D CNN only (3D CNN commented out)."""
    cmap = ListedColormap(PALETTE)
    oa_2d, aa_2d, kappa_2d = metrics_2d

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("HSI Classification — 2D CNN — Indian Pines", fontsize=14)

    # (a) Ground truth
    axes[0].imshow(gt, cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
    axes[0].set_title("(a) Ground truth")
    axes[0].axis("off")

    # (b) 2D CNN classification map
    axes[1].imshow(pred_2d, cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
    axes[1].set_title(f"(b) 2D CNN\nOA={oa_2d*100:.1f}%  Kappa={kappa_2d:.3f}")
    axes[1].axis("off")

    # (c) Training loss
    axes[2].plot(loss_2d, label="2D CNN", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("(c) Training loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Legend
    patches_legend = [
        mpatches.Patch(color=PALETTE[c], label=CLASS_NAMES[c])
        for c in range(1, 17)
    ]
    fig.legend(handles=patches_legend, loc="lower center", ncol=8,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = "./cnn_classification_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.show()

    # Detailed confusion matrix
    classes_present = np.unique(y_te)
    cm = confusion_matrix(y_te, y_pred_2d, labels=classes_present)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[CLASS_NAMES[c + 1] for c in classes_present],
    )
    fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax_cm, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax_cm.set_title("Confusion Matrix — 2D CNN")
    plt.tight_layout()
    cm_path = "./cnn_confusion_matrix_2d_cnn.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved → {cm_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load and preprocess
    cube, gt = load_data()
    cube_pca = preprocess(cube, N_PCA_COMPONENTS)

    # Extract patches
    patches, labels, coords = create_patches(cube_pca, gt, PATCH_SIZE)

    # Prepare data loaders
    (train_2d, test_2d), y_te = prepare_loaders(patches, labels)

    n_bands = N_PCA_COMPONENTS

    # ── Train 2D CNN ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training 2D CNN ...")
    print("=" * 60)
    model_2d = CNN2D(n_bands=n_bands, n_classes=NUM_CLASSES, patch_size=PATCH_SIZE)
    print(f"2D CNN parameters: {sum(p.numel() for p in model_2d.parameters()):,}")
    loss_2d = train_model(model_2d, train_2d)
    y_pred_2d = evaluate_model(model_2d, test_2d)
    metrics_2d = print_metrics(y_te, y_pred_2d, "2D CNN")

    # # ── Train 3D CNN ──────────────────────────────────────────────────────
    # print("\n" + "=" * 60)
    # print("Training 3D CNN ...")
    # print("=" * 60)
    # model_3d = CNN3D(n_bands=n_bands, n_classes=NUM_CLASSES, patch_size=PATCH_SIZE)
    # print(f"3D CNN parameters: {sum(p.numel() for p in model_3d.parameters()):,}")
    # loss_3d = train_model(model_3d, train_3d)
    # y_pred_3d = evaluate_model(model_3d, test_3d)
    # metrics_3d = print_metrics(y_te, y_pred_3d, "3D CNN")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'2D CNN':>10}")
    print("-" * 40)
    print(f"{'Overall Accuracy (OA)':<30} {metrics_2d[0]*100:>9.2f}%")
    print(f"{'Average Accuracy (AA)':<30} {metrics_2d[1]*100:>9.2f}%")
    print(f"{'Kappa Coefficient':<30} {metrics_2d[2]:>10.4f}")
    print("=" * 40)

    # ── Full classification maps ──────────────────────────────────────────
    print("\nGenerating full classification map (this may take a moment) ...")
    pred_map_2d = predict_full_map(model_2d, cube_pca, gt, PATCH_SIZE, mode="2d")
    # pred_map_3d = predict_full_map(model_3d, cube_pca, gt, PATCH_SIZE, mode="3d")

    # ── Visualize (2D CNN only) ───────────────────────────────────────────
    visualize_results_2d(gt, pred_map_2d, metrics_2d, loss_2d, y_te, y_pred_2d)


if __name__ == "__main__":
    main()
