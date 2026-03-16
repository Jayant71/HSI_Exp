"""
Hyperspectral Image Segmentation & Classification — Indian Pines Dataset
==========================================================================
Complete pipeline:
  1. Download Indian Pines data (HSI cube + ground truth)
  2. Preprocessing  — band removal, normalization
  3. Dimensionality reduction — PCA
  4. Segmentation  — SLIC superpixels
  5. Feature extraction — mean spectral signature per superpixel
  6. Classification — SVM (+ optional Random Forest)
  7. Evaluation    — OA, AA, Kappa, per-class accuracy
  8. Visualization — false-colour, PCA components, classification map

Requirements:
    pip install numpy scipy scikit-learn scikit-image matplotlib spectral requests
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import urllib.request
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from scipy.io import loadmat

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from skimage.segmentation import slic
from skimage.util import img_as_float


# ──────────────────────────────────────────────────────────────────────────────
# 1. Download Indian Pines dataset
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./IndianPines"
HSI_URL  = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
GT_URL   = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
HSI_FILE = os.path.join(DATA_DIR, "Indian_pines_corrected.mat")
GT_FILE  = os.path.join(DATA_DIR, "Indian_pines_gt.mat")

os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url: str, dest: str) -> None:
    if not os.path.exists(dest):
        print(f"Downloading {os.path.basename(dest)} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved → {dest}")
    else:
        print(f"  {os.path.basename(dest)} already present, skipping download.")

# download_file(HSI_URL, HSI_FILE)
# download_file(GT_URL,  GT_FILE)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Load data
# ──────────────────────────────────────────────────────────────────────────────
hsi_mat = loadmat(HSI_FILE)
gt_mat  = loadmat(GT_FILE)

# The corrected cube has water-absorption bands already removed → 200 bands
cube = hsi_mat["indian_pines_corrected"].astype(np.float32)   # (145, 145, 200)
gt   = gt_mat["indian_pines_gt"].astype(np.int32)              # (145, 145)

H, W, B = cube.shape
print(f"Cube shape : {cube.shape}  (rows × cols × bands)")
print(f"GT shape   : {gt.shape}")
print(f"Classes    : {np.unique(gt)}")   # 0 = background, 1–16 = land covers

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

# Colour palette (one per class 0–16)
PALETTE = np.array([
    [0,   0,   0  ],   # 0  background — black
    [54,  100, 27 ],   # 1  alfalfa
    [0,   168, 226],   # 2  corn-notill
    [76,  153, 0  ],   # 3  corn-mintill
    [255, 211, 0  ],   # 4  corn
    [148, 103, 189],   # 5  grass-pasture
    [23,  190, 207],   # 6  grass-trees
    [188, 189, 34 ],   # 7  grass-pasture-mowed
    [214, 39,  40 ],   # 8  hay-windrowed
    [31,  119, 180],   # 9  oats
    [255, 127, 14 ],   # 10 soybean-notill
    [44,  160, 44 ],   # 11 soybean-mintill
    [174, 199, 232],   # 12 soybean-clean
    [255, 152, 150],   # 13 wheat
    [197, 176, 213],   # 14 woods
    [196, 156, 148],   # 15 buildings-grass
    [247, 182, 210],   # 16 stone-steel
], dtype=np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing — normalize per band (zero-mean, unit-variance)
# ──────────────────────────────────────────────────────────────────────────────
cube_2d = cube.reshape(-1, B)          # (H*W, B)
scaler  = StandardScaler()
cube_2d_norm = scaler.fit_transform(cube_2d)   # still (H*W, B)
cube_norm = cube_2d_norm.reshape(H, W, B)
print("Preprocessing done — each band zero-mean, unit-variance.")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Dimensionality reduction — PCA → keep 99% variance (typically ~30 PCs)
# ──────────────────────────────────────────────────────────────────────────────
N_COMPONENTS = 30   # hard cap; PCA will use min(n_samples, n_features, this)
pca = PCA(n_components=N_COMPONENTS, whiten=False)
pc_2d = pca.fit_transform(cube_2d_norm)        # (H*W, N_COMPONENTS)
pc_cube = pc_2d.reshape(H, W, N_COMPONENTS)    # (H, W, N_COMPONENTS)

explained = np.cumsum(pca.explained_variance_ratio_)
# If 99% is not reached within the capped components, keep all available PCs.
n_keep = int(np.searchsorted(explained, 0.99, side="left")) + 1
n_keep = min(n_keep, explained.size)
print(f"PCA: {N_COMPONENTS} components computed, "
      f"{n_keep} explain ≥99% variance "
      f"({explained[n_keep-1]*100:.1f}%)")

# Use only the informative components downstream
pc_cube_k = pc_cube[:, :, :n_keep]
pc_2d_k   = pc_2d[:, :n_keep]


# ──────────────────────────────────────────────────────────────────────────────
# 5. SLIC superpixel segmentation
#    Applied to the first 3 PCs (treated as a pseudo-RGB image)
# ──────────────────────────────────────────────────────────────────────────────
N_SEGMENTS   = 200    # approximate number of superpixels
COMPACTNESS  = 0.1    # lower = more shape-irregular, spectrally tighter

pseudo_rgb = pc_cube[:, :, :3].copy()
# Rescale each channel to [0, 1] for SLIC
for c in range(3):
    ch = pseudo_rgb[:, :, c]
    pseudo_rgb[:, :, c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

segments = slic(
    pseudo_rgb,
    n_segments=N_SEGMENTS,
    compactness=COMPACTNESS,
    channel_axis=2,
    start_label=0,
)
n_sp = segments.max() + 1
print(f"SLIC produced {n_sp} superpixels.")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Feature extraction — mean PCA vector per superpixel
# ──────────────────────────────────────────────────────────────────────────────
# For each superpixel: mean of reduced features + majority GT label
sp_features = np.zeros((n_sp, n_keep), dtype=np.float32)
sp_labels   = np.zeros(n_sp, dtype=np.int32)

for sp_id in range(n_sp):
    mask = segments == sp_id
    sp_features[sp_id] = pc_2d_k[mask.ravel()].mean(axis=0)
    labels_in_sp = gt[mask]
    # Majority vote (ignore background=0 if possible)
    labels_fg = labels_in_sp[labels_in_sp > 0]
    if len(labels_fg) > 0:
        counts = np.bincount(labels_fg)
        sp_labels[sp_id] = counts.argmax()
    else:
        sp_labels[sp_id] = 0   # background superpixel

# Keep only labelled (non-background) superpixels for train/test split
valid_mask   = sp_labels > 0
X_valid      = sp_features[valid_mask]
y_valid      = sp_labels[valid_mask]
valid_sp_ids = np.where(valid_mask)[0]

print(f"Labelled superpixels: {len(y_valid)} / {n_sp}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Classification — SVM with RBF kernel
# ──────────────────────────────────────────────────────────────────────────────
TEST_SIZE   = 0.3    # 30% superpixels held out for testing
RANDOM_SEED = 42

class_counts = np.bincount(y_valid)
min_class_count = class_counts[class_counts > 0].min()
stratify_labels = y_valid if min_class_count >= 2 else None
if stratify_labels is None:
    print(
        "Warning: At least one class has fewer than 2 superpixels; "
        "using non-stratified train/test split."
    )

X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X_valid, y_valid, valid_sp_ids,
    test_size=TEST_SIZE, stratify=stratify_labels, random_state=RANDOM_SEED
)

print(f"\nTraining on {len(y_tr)} superpixels, testing on {len(y_te)}.")

# ── SVM ────────────────────────────────────────────────────────────────────
svm = SVC(kernel="rbf", C=100, gamma="scale", random_state=RANDOM_SEED)
svm.fit(X_tr, y_tr)
y_pred_svm = svm.predict(X_te)

oa_svm    = accuracy_score(y_te, y_pred_svm)
kappa_svm = cohen_kappa_score(y_te, y_pred_svm)
print(f"\n[SVM]  OA = {oa_svm*100:.2f}%  |  Kappa = {kappa_svm:.4f}")

# Per-class accuracy (AA)
classes_present = np.unique(y_te)
per_class_acc = []
for c in classes_present:
    mask_c = y_te == c
    per_class_acc.append(accuracy_score(y_te[mask_c], y_pred_svm[mask_c]))
aa_svm = np.mean(per_class_acc)
print(f"[SVM]  AA = {aa_svm*100:.2f}%")

print("\nPer-class report (SVM):")
print(classification_report(
    y_te, y_pred_svm,
    labels=classes_present,
    target_names=[CLASS_NAMES[c] for c in classes_present],
    zero_division=0,
))

# ── Random Forest (bonus) ──────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED)
rf.fit(X_tr, y_tr)
y_pred_rf = rf.predict(X_te)

oa_rf    = accuracy_score(y_te, y_pred_rf)
kappa_rf = cohen_kappa_score(y_te, y_pred_rf)
print(f"\n[RF]   OA = {oa_rf*100:.2f}%  |  Kappa = {kappa_rf:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Build full classification map (predict ALL superpixels)
# ──────────────────────────────────────────────────────────────────────────────
all_preds_svm = np.zeros(n_sp, dtype=np.int32)
all_preds_svm[valid_sp_ids] = svm.predict(X_valid)

# Map superpixel predictions back to pixel space
pred_map_svm = np.zeros((H, W), dtype=np.int32)
for sp_id in range(n_sp):
    pred_map_svm[segments == sp_id] = all_preds_svm[sp_id]


# ──────────────────────────────────────────────────────────────────────────────
# 9. Visualization
# ──────────────────────────────────────────────────────────────────────────────
cmap = ListedColormap(PALETTE)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Hyperspectral Segmentation & Classification — Indian Pines", fontsize=14)

# (a) False-colour composite (band 29, 19, 9 → R, G, B)
def stretch(arr):
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

fc = np.stack([stretch(cube[:, :, 29]),
               stretch(cube[:, :, 19]),
               stretch(cube[:, :, 9])], axis=2)
axes[0, 0].imshow(fc)
axes[0, 0].set_title("(a) False-colour composite")
axes[0, 0].axis("off")

# (b) Ground truth
axes[0, 1].imshow(gt, cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
axes[0, 1].set_title("(b) Ground truth labels")
axes[0, 1].axis("off")

# (c) PCA — first component
pc1 = pc_cube[:, :, 0]
axes[0, 2].imshow(pc1, cmap="viridis")
axes[0, 2].set_title("(c) PCA — component 1")
axes[0, 2].axis("off")

# (d) SLIC superpixel boundaries
from skimage.segmentation import mark_boundaries
axes[1, 0].imshow(mark_boundaries(fc, segments, color=(1, 1, 0)))
axes[1, 0].set_title(f"(d) SLIC superpixels ({n_sp})")
axes[1, 0].axis("off")

# (e) SVM classification map
axes[1, 1].imshow(pred_map_svm, cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
axes[1, 1].set_title(f"(e) SVM classification map\nOA={oa_svm*100:.1f}%  Kappa={kappa_svm:.3f}")
axes[1, 1].axis("off")

# (f) PCA explained variance
axes[1, 2].plot(np.cumsum(pca.explained_variance_ratio_) * 100, color="#185FA5", linewidth=1.5)
axes[1, 2].axhline(99, color="#D85A30", linestyle="--", linewidth=1, label="99% threshold")
axes[1, 2].axvline(n_keep - 1, color="#639922", linestyle="--", linewidth=1,
                   label=f"n={n_keep} components")
axes[1, 2].set_xlabel("Number of components")
axes[1, 2].set_ylabel("Cumulative explained variance (%)")
axes[1, 2].set_title("(f) PCA explained variance")
axes[1, 2].legend(fontsize=9)
axes[1, 2].grid(True, alpha=0.3)

# Legend below the figure
patches = [
    mpatches.Patch(color=PALETTE[c], label=CLASS_NAMES[c])
    for c in range(1, 17)
]
fig.legend(
    handles=patches,
    loc="lower center",
    ncol=8,
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04),
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
out_path = "./indian_pines_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out_path}")
plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 10. Confusion matrix (SVM)
# ──────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_te, y_pred_svm, labels=classes_present)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[CLASS_NAMES[c] for c in classes_present],
)
fig2, ax2 = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax2, xticks_rotation=45, colorbar=True, cmap="Blues")
ax2.set_title("Confusion Matrix — SVM (superpixel level)")
plt.tight_layout()
cm_path = "./indian_pines_confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
print(f"Confusion matrix saved → {cm_path}")
plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 11. Summary metrics table
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
print(f"{'Metric':<30} {'SVM':>10} {'RF':>10}")
print("-"*50)
print(f"{'Overall Accuracy (OA)':<30} {oa_svm*100:>9.2f}% {oa_rf*100:>9.2f}%")
print(f"{'Average Accuracy (AA)':<30} {aa_svm*100:>9.2f}% {'N/A':>10}")
print(f"{'Kappa Coefficient':<30} {kappa_svm:>10.4f} {kappa_rf:>10.4f}")
print("="*50)