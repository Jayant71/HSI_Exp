import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# Default class names for supported datasets
DATASET_CLASS_NAMES = {
    "ip": [
        "Background", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
        "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
        "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
        "Stone-Steel-Towers",
    ],
    "pu": [
        "Background", "Asphalt", "Meadows", "Gravel", "Trees",
        "Painted metal sheets", "Bare Soil", "Bitumen",
        "Self-Blocking Bricks", "Shadows",
    ],
    "sa": [
        "Background", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2",
        "Fallow", "Fallow_rough_plow", "Fallow_smooth", "Stubble",
        "Celery", "Grapes_untrained", "Soil_vinyard_develop",
        "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
        "Lettuce_romaine_5wk", "Lettuce_romaine_6wk",
        "Lettuce_romaine_7wk", "Vinyard_untrained",
        "Vinyard_vertical_trellis",
    ],
}


def get_class_colormap(n_classes):
    """Return a ListedColormap with *n_classes* distinct colours."""
    base = plt.cm.get_cmap("tab20", n_classes)
    colors = base(np.linspace(0, 1, n_classes))
    colors[0] = [0, 0, 0, 1]  # black for background
    return ListedColormap(colors)


# ------------------------------------------------------------------
# Ground-truth / classification map
# ------------------------------------------------------------------

def plot_ground_truth(gt, dataset="ip", title="Ground Truth", ax=None,
                      figsize=(8, 8), save_path=None):
    """Display the ground-truth classification map with a legend."""
    n_classes = int(gt.max()) + 1
    cmap = get_class_colormap(n_classes)
    class_names = DATASET_CLASS_NAMES.get(dataset, [str(i) for i in range(n_classes)])

    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(gt, cmap=cmap, interpolation="nearest", vmin=0, vmax=n_classes - 1)
    ax.set_title(title)
    ax.axis("off")

    patches = [
        mpatches.Patch(color=cmap(i), label=class_names[i])
        for i in range(min(n_classes, len(class_names)))
        if i == 0 or np.any(gt == i)
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left",
              fontsize="small", frameon=False)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.tight_layout()
        plt.show()


def plot_classification_map(prediction, gt, dataset="ip",
                            figsize=(16, 8), save_path=None):
    """Show ground truth and predicted classification maps side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plot_ground_truth(gt, dataset=dataset, title="Ground Truth", ax=ax1)
    plot_ground_truth(prediction, dataset=dataset, title="Predicted", ax=ax2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# Spectral signatures
# ------------------------------------------------------------------

def plot_spectral_signature(cube, gt, class_ids=None, dataset="ip",
                            figsize=(10, 5), save_path=None):
    """Plot mean spectral signature for selected classes.

    Parameters
    ----------
    cube : ndarray (H, W, B) -- raw or normalised hyperspectral cube.
    gt   : ndarray (H, W)    -- ground-truth labels.
    class_ids : list[int] | None -- classes to plot; ``None`` plots all.
    """
    class_names = DATASET_CLASS_NAMES.get(dataset, None)
    if class_ids is None:
        class_ids = [c for c in np.unique(gt) if c != 0]

    fig, ax = plt.subplots(figsize=figsize)
    for cid in class_ids:
        mask = gt == cid
        spectra = cube[mask]
        mean_spectrum = spectra.mean(axis=0)
        std_spectrum = spectra.std(axis=0)
        bands = np.arange(1, len(mean_spectrum) + 1)
        label = class_names[cid] if class_names and cid < len(class_names) else f"Class {cid}"
        ax.plot(bands, mean_spectrum, label=label)
        ax.fill_between(bands, mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum, alpha=0.15)

    ax.set_xlabel("Band Index")
    ax.set_ylabel("Reflectance / Intensity")
    ax.set_title("Mean Spectral Signatures")
    ax.legend(fontsize="small", loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_pixel_spectrum(cube, row=None, col=None, pixels=None,
                        labels=None, figsize=(8, 4), save_path=None):
    """Plot the spectrum of one or more pixels.

    Parameters
    ----------
    cube   : ndarray of shape (H, W, B).
    row, col : int -- single pixel coordinates (kept for backward compat).
    pixels : list[tuple[int, int]] -- list of (row, col) pairs to plot.
             If provided, *row* and *col* are ignored.
    labels : list[str] | None -- legend label for each pixel.
    """
    if pixels is None:
        if row is None or col is None:
            raise ValueError("Provide either (row, col) or pixels=[(r,c), ...]")
        pixels = [(row, col)]

    n_bands = cube.shape[2]
    bands = np.arange(1, n_bands + 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (r, c) in enumerate(pixels):
        spectrum = cube[r, c, :]
        label = labels[i] if labels and i < len(labels) else f"({r}, {c})"
        ax.plot(bands, spectrum, linewidth=1.2, label=label)

    ax.set_xlabel("Band Index")
    ax.set_ylabel("Reflectance / Intensity")
    ax.set_title("Pixel Spectra" if len(pixels) > 1 else f"Pixel Spectrum at {pixels[0]}")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# Band visualisation
# ------------------------------------------------------------------

def plot_band(cube, band_index, title=None, cmap="gray",
              figsize=(6, 6), ax=None, save_path=None):
    """Display a single spectral band as a grayscale image."""
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(cube[:, :, band_index], cmap=cmap)
    ax.set_title(title or f"Band {band_index}")
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.tight_layout()
        plt.show()


def plot_band_grid(cube, band_indices=None, n_bands=16, cols=4,
                   cmap="gray", figsize=(14, 14), save_path=None):
    """Show a grid of spectral bands.

    Parameters
    ----------
    band_indices : list[int] | None -- specific bands to show.
    n_bands      : int -- number of evenly-spaced bands when *band_indices* is None.
    """
    total_bands = cube.shape[2]
    if band_indices is None:
        band_indices = np.linspace(0, total_bands - 1, n_bands, dtype=int)

    n = len(band_indices)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.asarray(axes).flatten()

    for i, b in enumerate(band_indices):
        axes[i].imshow(cube[:, :, b], cmap=cmap)
        axes[i].set_title(f"Band {b}", fontsize=9)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Spectral Band Grid", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_rgb_composite(cube, r=29, g=19, b=9, figsize=(6, 6),
                       stretch_pct=(2, 98), save_path=None):
    """Display a false-colour RGB composite from three bands.

    Parameters
    ----------
    r, g, b       : int -- band indices for the R, G, B channels.
    stretch_pct   : tuple -- percentile stretch for contrast enhancement.
    """
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1).astype(np.float64)
    lo, hi = np.percentile(rgb, stretch_pct)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)
    ax.set_title(f"RGB Composite (R={r}, G={g}, B={b})")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# PCA / dimensionality-reduction helpers
# ------------------------------------------------------------------

def plot_pca_variance(cube, n_components=None, figsize=(8, 4),
                      save_path=None):
    """Plot cumulative explained variance from PCA of the cube."""
    H, W, B = cube.shape
    if n_components is None:
        n_components = B
    flat = cube.reshape(-1, B).astype(np.float64)
    flat -= flat.mean(axis=0)
    cov = np.cov(flat, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.bar(range(1, n_components + 1), explained[:n_components])
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Per-component Variance")

    ax2.plot(range(1, n_components + 1), cumulative[:n_components], marker="o", markersize=3)
    ax2.axhline(y=0.99, color="r", linestyle="--", label="99 %")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_pca_bands(cube_pca, n_show=3, figsize=(14, 4), save_path=None):
    """Visualise the first *n_show* PCA components as images."""
    fig, axes = plt.subplots(1, n_show, figsize=figsize)
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        axes[i].imshow(cube_pca[:, :, i], cmap="viridis")
        axes[i].set_title(f"PC {i + 1}")
        axes[i].axis("off")
    plt.suptitle("Principal Components", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# Histogram / distribution helpers
# ------------------------------------------------------------------

def plot_class_distribution(gt, dataset="ip", figsize=(10, 5),
                            save_path=None):
    """Bar chart showing the number of pixels per class."""
    class_names = DATASET_CLASS_NAMES.get(dataset, None)
    classes, counts = np.unique(gt, return_counts=True)

    labels = []
    plot_counts = []
    for c, cnt in zip(classes, counts):
        if c == 0:
            continue
        labels.append(class_names[c] if class_names and c < len(class_names) else f"Class {c}")
        plot_counts.append(cnt)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, plot_counts)
    ax.set_xlabel("Pixel Count")
    ax.set_title("Class Distribution")
    ax.bar_label(bars, padding=3, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_band_histogram(cube, band_indices=None, n_bands=5,
                        figsize=(10, 5), save_path=None):
    """Overlay histograms for selected bands."""
    total_bands = cube.shape[2]
    if band_indices is None:
        band_indices = np.linspace(0, total_bands - 1, n_bands, dtype=int)

    fig, ax = plt.subplots(figsize=figsize)
    for b in band_indices:
        ax.hist(cube[:, :, b].ravel(), bins=100, alpha=0.5, label=f"Band {b}")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Band Histograms")
    ax.legend(fontsize="small")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# Correlation & statistics
# ------------------------------------------------------------------

def plot_band_correlation(cube, figsize=(8, 7), save_path=None):
    """Display the band-to-band correlation matrix as a heatmap."""
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    corr = np.corrcoef(flat, rowvar=False)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("Band")
    ax.set_ylabel("Band")
    ax.set_title("Band Correlation Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def plot_training_curves(history, figsize=(12, 4), save_path=None):
    """Plot loss and accuracy curves from a training history dict.

    Parameters
    ----------
    history : dict with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if "loss" in history:
        ax1.plot(history["loss"], label="Train Loss")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if "accuracy" in history:
        ax2.plot(history["accuracy"], label="Train Acc")
    if "val_accuracy" in history:
        ax2.plot(history["val_accuracy"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False,
                          figsize=(10, 8), cmap="Blues", save_path=None):
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    normalize : bool -- if True, show percentages instead of counts.
    """
    labels = sorted(set(y_true) | set(y_pred))
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.where(row_sums > 0, cm / row_sums, 0)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    if class_names is None:
        class_names = [str(l) for l in labels]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm_display.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = f"{cm_display[i, j]:{fmt}}"
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black",
                    fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
