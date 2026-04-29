"""Cluster stability validation via Hennig-style cluster-wise Jaccard similarity.

Reference: Hennig C (2007) Cluster-wise assessment of cluster stability.
Computational Statistics & Data Analysis 52(1):258-271.

Method
------
1. Fit the chosen algorithm on the full cohort → reference labels.
2. For each of `n_iter` iterations:
   a. Sample 80% of the data (without replacement).
   b. Refit the same algorithm (same K) on the subsample.
   c. For each reference cluster c, find the subsample cluster s* that maximises
      Jaccard(c ∩ subsample, s*).
   d. Record that Jaccard score for cluster c.
3. Report mean Jaccard per reference cluster across all iterations.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

log = logging.getLogger(__name__)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _max_jaccard_for_cluster(ref_mask: np.ndarray,
                              sub_idx: np.ndarray,
                              sub_labels: np.ndarray) -> float:
    """Return the best Jaccard between one reference cluster and any subsample cluster.

    Parameters
    ----------
    ref_mask   : boolean array (full cohort size) — True for members of the reference cluster
    sub_idx    : 1-D array of full-cohort indices that were sampled (ordered as in X_sub)
    sub_labels : labels (subsample size) aligned to sub_idx order
    """
    # Build map: full-cohort index → subsample row position (0..n_sub-1)
    # sub_idx[i] is the full-cohort index of the i-th subsample row
    full_to_sub = {fi: si for si, fi in enumerate(sub_idx)}

    # Subsample row positions of the reference cluster's members
    ref_in_sub = {full_to_sub[fi] for fi in np.where(ref_mask)[0] if fi in full_to_sub}
    if not ref_in_sub:
        return 0.0

    # Find the subsample cluster with maximum Jaccard overlap
    best = 0.0
    for k in np.unique(sub_labels):
        sub_cluster = set(np.where(sub_labels == k)[0])
        j = _jaccard(ref_in_sub, sub_cluster)
        if j > best:
            best = j
    return best


def _make_fitter(algorithm: str, k: int,
                 random_state: int) -> Callable[[np.ndarray], np.ndarray]:
    if algorithm == "kmeans":
        def fit(X):
            return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
    elif algorithm == "gmm":
        def fit(X):
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  n_init=3, random_state=random_state)
            gmm.fit(X)
            return gmm.predict(X)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return fit


def cluster_stability(
    X: np.ndarray,
    ref_labels: np.ndarray,
    algorithm: str,
    k: int,
    n_iter: int = 20,
    subsample_frac: float = 0.80,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compute Hennig-style cluster-wise Jaccard stability scores.

    Parameters
    ----------
    X            : preprocessed feature matrix (n_samples, n_features)
    ref_labels   : cluster assignments on the full cohort
    algorithm    : "kmeans" or "gmm"
    k            : number of clusters
    n_iter       : number of subsample iterations (default 20)
    subsample_frac : fraction of data to use per iteration (default 0.80)
    random_state : seed for reproducibility

    Returns
    -------
    DataFrame with columns [cluster, mean_jaccard, std_jaccard, n_stable_iters]
    where n_stable_iters = number of iters with Jaccard > 0.5.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    n_sub = int(n * subsample_frac)
    unique_clusters = np.unique(ref_labels)

    # Accumulate per-cluster Jaccard scores across iterations
    jaccard_records: dict[int, list[float]] = {c: [] for c in unique_clusters}

    fitter = _make_fitter(algorithm, k, random_state=random_state)

    for it in range(n_iter):
        sub_idx = rng.choice(n, size=n_sub, replace=False)
        X_sub = X[sub_idx]
        try:
            sub_labels = fitter(X_sub)
        except Exception as e:
            log.warning("Iteration %d failed: %s", it, e)
            continue

        for c in unique_clusters:
            ref_mask = ref_labels == c
            j = _max_jaccard_for_cluster(ref_mask, sub_idx, sub_labels)
            jaccard_records[c].append(j)

        if (it + 1) % 5 == 0:
            log.info("Stability: completed %d/%d iterations", it + 1, n_iter)

    rows = []
    for c in unique_clusters:
        scores = np.array(jaccard_records[c])
        rows.append({
            "cluster": int(c),
            "mean_jaccard": float(np.mean(scores)) if len(scores) else 0.0,
            "std_jaccard": float(np.std(scores)) if len(scores) else 0.0,
            "n_stable_iters": int(np.sum(scores > 0.5)),
            "n_iters": len(scores),
        })

    result = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    result["stability_tier"] = result["mean_jaccard"].map(
        lambda j: "Highly stable (≥0.85)" if j >= 0.85
        else ("Stable (0.60–0.84)" if j >= 0.60 else "Unstable (<0.60)")
    )
    return result
