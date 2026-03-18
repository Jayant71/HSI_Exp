import numpy as np


# ------------------------------------------------------------------
# Core classification metrics
# ------------------------------------------------------------------

def confusion_matrix(y_true, y_pred):
    """Compute a confusion matrix (rows=true, cols=predicted).

    Returns
    -------
    cm     : ndarray (n_classes, n_classes)
    labels : sorted list of unique class labels
    """
    labels = sorted(set(y_true) | set(y_pred))
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels


def overall_accuracy(y_true, y_pred):
    """Overall Accuracy (OA) -- fraction of correctly classified samples."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def per_class_accuracy(y_true, y_pred):
    """Per-class accuracy (producer's accuracy / recall per class).

    Returns
    -------
    acc_dict : dict  {class_label: accuracy}
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    result = {}
    for i, lbl in enumerate(labels):
        total = cm[i].sum()
        result[lbl] = cm[i, i] / total if total > 0 else 0.0
    return result


def average_accuracy(y_true, y_pred):
    """Average Accuracy (AA) -- mean of per-class accuracies."""
    acc = per_class_accuracy(y_true, y_pred)
    return np.mean(list(acc.values()))


def precision(y_true, y_pred, average="macro"):
    """Precision (per-class or macro-averaged).

    Parameters
    ----------
    average : 'macro' | 'micro' | None
        None returns a dict of per-class values.
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    col_sums = cm.sum(axis=0)
    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = cm[i, i] / col_sums[i] if col_sums[i] > 0 else 0.0

    if average is None:
        return per_class
    if average == "micro":
        return np.trace(cm) / cm.sum()
    return np.mean(list(per_class.values()))


def recall(y_true, y_pred, average="macro"):
    """Recall (per-class or macro-averaged)."""
    cm, labels = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)
    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[lbl] = cm[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0

    if average is None:
        return per_class
    if average == "micro":
        return np.trace(cm) / cm.sum()
    return np.mean(list(per_class.values()))


def f1_score(y_true, y_pred, average="macro"):
    """F1 score (per-class or macro-averaged).

    Parameters
    ----------
    average : 'macro' | 'micro' | None
    """
    p = precision(y_true, y_pred, average=None)
    r = recall(y_true, y_pred, average=None)
    per_class = {}
    for lbl in p:
        denom = p[lbl] + r[lbl]
        per_class[lbl] = (2 * p[lbl] * r[lbl]) / denom if denom > 0 else 0.0

    if average is None:
        return per_class
    if average == "micro":
        mp = precision(y_true, y_pred, average="micro")
        mr = recall(y_true, y_pred, average="micro")
        return (2 * mp * mr) / (mp + mr) if (mp + mr) > 0 else 0.0
    return np.mean(list(per_class.values()))


# ------------------------------------------------------------------
# Kappa coefficient
# ------------------------------------------------------------------

def kappa(y_true, y_pred):
    """Cohen's Kappa coefficient -- agreement corrected for chance."""
    cm, _ = confusion_matrix(y_true, y_pred)
    n = cm.sum()
    po = np.trace(cm) / n  # observed agreement
    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n * n)  # expected
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0


# ------------------------------------------------------------------
# Convenience: full classification report
# ------------------------------------------------------------------

def classification_report(y_true, y_pred, class_names=None):
    """Print a classification report similar to sklearn.

    Returns
    -------
    report : dict  with per-class and aggregate metrics.
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    p = precision(y_true, y_pred, average=None)
    r = recall(y_true, y_pred, average=None)
    f = f1_score(y_true, y_pred, average=None)
    support = {lbl: int(cm[i].sum()) for i, lbl in enumerate(labels)}

    report = {}
    header = f"{'Class':>30s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'Support':>8s}"
    print(header)
    print("-" * len(header))
    for lbl in labels:
        name = class_names[lbl] if class_names and lbl < len(class_names) else str(lbl)
        print(f"{name:>30s}  {p[lbl]:6.4f}  {r[lbl]:6.4f}  {f[lbl]:6.4f}  {support[lbl]:8d}")
        report[lbl] = {"precision": p[lbl], "recall": r[lbl], "f1": f[lbl], "support": support[lbl]}

    oa = overall_accuracy(y_true, y_pred)
    aa = average_accuracy(y_true, y_pred)
    k = kappa(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("-" * len(header))
    print(f"{'OA':>30s}  {oa:6.4f}")
    print(f"{'AA':>30s}  {aa:6.4f}")
    print(f"{'Kappa':>30s}  {k:6.4f}")
    print(f"{'Macro F1':>30s}  {macro_f1:6.4f}")

    report["overall_accuracy"] = oa
    report["average_accuracy"] = aa
    report["kappa"] = k
    report["macro_f1"] = macro_f1
    return report


# ------------------------------------------------------------------
# Spectral / data-quality metrics
# ------------------------------------------------------------------

def snr(cube):
    """Estimate per-band Signal-to-Noise Ratio (SNR) of the HSI cube.

    SNR = mean / std  for each band.

    Returns
    -------
    snr_values : ndarray (B,)
    """
    H, W, B = cube.shape
    flat = cube.reshape(-1, B).astype(np.float64)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0) + 1e-10
    return means / stds


def spectral_angle(s1, s2):
    """Spectral Angle Mapper (SAM) between two spectra (in radians).

    Smaller angle = more similar.
    """
    s1, s2 = np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)
    cos_theta = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-10)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def sam_map(cube, reference_spectrum):
    """Compute a Spectral Angle Map for every pixel against a reference.

    Parameters
    ----------
    cube               : ndarray (H, W, B)
    reference_spectrum : ndarray (B,)

    Returns
    -------
    angle_map : ndarray (H, W) -- angle in radians.
    """
    H, W, B = cube.shape
    flat = cube.reshape(-1, B).astype(np.float64)
    ref = np.asarray(reference_spectrum, dtype=np.float64)
    dots = flat @ ref
    norms = np.linalg.norm(flat, axis=1) * np.linalg.norm(ref) + 1e-10
    cos_theta = np.clip(dots / norms, -1.0, 1.0)
    return np.arccos(cos_theta).reshape(H, W)


def spectral_information_divergence(s1, s2):
    """Spectral Information Divergence (SID) between two spectra.

    Uses KL-divergence: SID = KL(p||q) + KL(q||p).
    """
    eps = 1e-10
    p = np.asarray(s1, dtype=np.float64)
    q = np.asarray(s2, dtype=np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    return kl_pq + kl_qp


# ------------------------------------------------------------------
# Dataset statistics
# ------------------------------------------------------------------

def dataset_summary(cube, gt):
    """Print and return basic dataset statistics.

    Returns
    -------
    info : dict
    """
    H, W, B = cube.shape
    classes = np.unique(gt)
    n_classes = len(classes[classes != 0])
    labelled = int(np.sum(gt > 0))

    info = {
        "height": H,
        "width": W,
        "bands": B,
        "n_classes": n_classes,
        "total_pixels": H * W,
        "labelled_pixels": labelled,
        "unlabelled_pixels": H * W - labelled,
        "class_labels": classes.tolist(),
        "pixel_value_range": (float(cube.min()), float(cube.max())),
    }

    print(f"Cube shape      : {H} x {W} x {B}")
    print(f"Classes          : {n_classes} (labels: {classes[classes != 0].tolist()})")
    print(f"Labelled pixels  : {labelled} / {H * W} ({100 * labelled / (H * W):.1f} %)")
    print(f"Value range      : [{cube.min():.4f}, {cube.max():.4f}]")
    return info


def class_pixel_counts(gt):
    """Return a dict mapping each non-zero class label to its pixel count."""
    classes, counts = np.unique(gt, return_counts=True)
    return {int(c): int(cnt) for c, cnt in zip(classes, counts) if c != 0}
