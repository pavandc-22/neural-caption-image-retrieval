import numpy as np


def ranks_from_similarity(sim, gt):
    """sim: (Q, N) similarity, gt: (Q,) index of the correct image. Returns 1-based rank per query."""
    pos = sim[np.arange(len(gt)), gt][:, None]
    # pessimistic tie handling: every image scoring >= the correct one counts (itself included) -> rank >= 1.
    # Matters for order embeddings, where many images can have an identical zero penalty.
    return (sim >= pos).sum(1)


def summarize(ranks, ks=(1, 5, 10)):
    out = {f"R@{k}": float((ranks <= k).mean() * 100) for k in ks}
    out["Mean r"] = float(ranks.mean())
    out["Median r"] = float(np.median(ranks))
    return out


def evaluate_folds(sim_fn, n_images, caps_per_img=5, fold_size=1000, ks=(1, 5, 10)):
    """Report protocol (1K-image galleries, Karpathy-style): split the test set into folds of 1000 images;
    every caption of the fold's images is a query against the fold's 1000 images; average over folds.
    sim_fn(lo, hi) -> (queries=(hi-lo)*caps_per_img, gallery=hi-lo) similarity, queries ordered image-major.
    Returns (mean metrics, concatenated per-query ranks)."""
    all_ranks, per_fold = [], []
    for lo in range(0, n_images - fold_size + 1, fold_size):
        hi = lo + fold_size
        sim = sim_fn(lo, hi)
        gt = np.repeat(np.arange(fold_size), sim.shape[0] // fold_size)
        r = ranks_from_similarity(sim, gt)
        all_ranks.append(r)
        per_fold.append(summarize(r, ks))
    if not per_fold:                       # fewer than 1000 images: single fold of what we have
        sim = sim_fn(0, n_images)
        gt = np.repeat(np.arange(n_images), sim.shape[0] // n_images)
        r = ranks_from_similarity(sim, gt)
        return summarize(r, ks), r
    mean = {k: float(np.mean([f[k] for f in per_fold])) for k in per_fold[0]}
    return mean, np.concatenate(all_ranks)
