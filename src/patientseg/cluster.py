"""Clustering pipeline: preprocessing, K-means, GMM, and model comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger(__name__)

# 2009 features used in clustering (outcome columns excluded)
CLUSTER_NUMERIC_COLS = [
    "n_chronic_conditions",
    "charlson_index",
    "n_unique_drugs_2009",
    "esrd_indicator",
    "ip_admissions_2009",
    "ip_days_2009",
    "ed_visits_2009",
    "op_visits_2009",
    "carrier_visits_2009",
    "total_cost_2009",
    "age_at_2010",
    "dual_eligible_months_2009",
]

CLUSTER_CATEGORICAL_COLS = ["sex", "race"]

# CCW binary flags — included as numeric (0/1)
CCW_BINARY_COLS = [
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
    "SP_RA_OA", "SP_STRKETIA",
]

# Columns to log1p-transform before scaling (count/cost features)
LOG_TRANSFORM_COLS = [
    "ip_admissions_2009", "ip_days_2009", "ed_visits_2009",
    "op_visits_2009", "carrier_visits_2009", "total_cost_2009",
    "n_unique_drugs_2009",
]


@dataclass
class ClusterResult:
    algorithm: str          # "kmeans" or "gmm"
    k: int
    labels: np.ndarray
    silhouette: float
    davies_bouldin: float
    bic: float | None       # GMM only
    inertia: float | None   # K-means only
    cost_f_stat: float      # ANOVA F-stat of 2010 cost across clusters


@dataclass
class ComparisonTable:
    kmeans: ClusterResult
    gmm: ClusterResult
    ari: float              # Adjusted Rand Index between the two solutions
    chosen: str             # "kmeans" or "gmm"
    justification: str

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in [self.kmeans, self.gmm]:
            rows.append({
                "Algorithm": r.algorithm.upper(),
                "K": r.k,
                "Silhouette": round(r.silhouette, 3),
                "Davies-Bouldin": round(r.davies_bouldin, 3),
                "BIC": round(r.bic, 1) if r.bic is not None else None,
                "Inertia": round(r.inertia, 1) if r.inertia is not None else None,
                "Cost F-stat (2010)": round(r.cost_f_stat, 1),
            })
        return pd.DataFrame(rows)


def _log1p_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))
    return df


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer for the clustering feature matrix."""
    numeric_cols = [c for c in CLUSTER_NUMERIC_COLS + CCW_BINARY_COLS if c in feature_df.columns]
    cat_cols = [c for c in CLUSTER_CATEGORICAL_COLS if c in feature_df.columns]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def preprocess(feature_df: pd.DataFrame) -> tuple[np.ndarray, ColumnTransformer]:
    """Log1p-transform skewed cols, then standardize + one-hot-encode.

    Returns
    -------
    X : np.ndarray  shape (n_beneficiaries, n_features)
    preprocessor : fitted ColumnTransformer
    """
    df = _log1p_transform(feature_df, LOG_TRANSFORM_COLS)
    preprocessor = build_preprocessor(df)
    X = preprocessor.fit_transform(df)
    log.info("Preprocessed feature matrix: %s", X.shape)
    return X, preprocessor


def _anova_f(labels: np.ndarray, cost_2010: np.ndarray) -> float:
    groups = [cost_2010[labels == k] for k in np.unique(labels)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return 0.0
    stat, _ = f_oneway(*groups)
    return float(stat) if np.isfinite(stat) else 0.0


def fit_kmeans_sweep(X: np.ndarray, cost_2010: np.ndarray,
                     k_range: range = range(2, 11),
                     random_state: int = 42) -> dict[int, ClusterResult]:
    """Fit K-means for each K in k_range. Returns dict {k: ClusterResult}."""
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(10000, len(labels)))
        db = davies_bouldin_score(X, labels)
        f = _anova_f(labels, cost_2010)
        results[k] = ClusterResult(
            algorithm="kmeans", k=k, labels=labels,
            silhouette=sil, davies_bouldin=db,
            bic=None, inertia=km.inertia_, cost_f_stat=f,
        )
        log.info("K-means k=%d | sil=%.3f | DB=%.3f | F=%.1f", k, sil, db, f)
    return results


def fit_gmm_sweep(X: np.ndarray, cost_2010: np.ndarray,
                  k_range: range = range(2, 11),
                  random_state: int = 42) -> dict[int, ClusterResult]:
    """Fit GMM for each K in k_range. Returns dict {k: ClusterResult}."""
    results = {}
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=5, random_state=random_state)
        gmm.fit(X)
        labels = gmm.predict(X)
        sil = silhouette_score(X, labels, sample_size=min(10000, len(labels)))
        db = davies_bouldin_score(X, labels)
        f = _anova_f(labels, cost_2010)
        results[k] = ClusterResult(
            algorithm="gmm", k=k, labels=labels,
            silhouette=sil, davies_bouldin=db,
            bic=gmm.bic(X), inertia=None, cost_f_stat=f,
        )
        log.info("GMM k=%d | sil=%.3f | DB=%.3f | BIC=%.1f | F=%.1f", k, sil, db, gmm.bic(X), f)
    return results


def pick_k_kmeans(sweep: dict[int, ClusterResult], min_k: int = 4) -> int:
    """Pick K for K-means: highest silhouette at or above min_k.

    min_k guards against trivially low K values (e.g., K=2) that are
    algorithmically optimal but yield few clinically actionable segments.
    """
    eligible = {k: v for k, v in sweep.items() if k >= min_k}
    if eligible:
        return max(eligible, key=lambda k: eligible[k].silhouette)
    return max(sweep, key=lambda k: sweep[k].silhouette)


def pick_k_gmm(sweep: dict[int, ClusterResult]) -> int:
    """Pick K for GMM: lowest BIC."""
    return min(sweep, key=lambda k: sweep[k].bic)


def refit_kmeans(X: np.ndarray, k: int, cost_2010: np.ndarray,
                 random_state: int = 42) -> tuple[KMeans, ClusterResult]:
    km = KMeans(n_clusters=k, n_init=50, random_state=random_state)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(10000, len(labels)))
    db = davies_bouldin_score(X, labels)
    f = _anova_f(labels, cost_2010)
    return km, ClusterResult("kmeans", k, labels, sil, db, None, km.inertia_, f)


def refit_gmm(X: np.ndarray, k: int, cost_2010: np.ndarray,
              random_state: int = 42) -> tuple[GaussianMixture, ClusterResult]:
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=10, random_state=random_state)
    gmm.fit(X)
    labels = gmm.predict(X)
    sil = silhouette_score(X, labels, sample_size=min(10000, len(labels)))
    db = davies_bouldin_score(X, labels)
    f = _anova_f(labels, cost_2010)
    return gmm, ClusterResult("gmm", k, labels, sil, db, gmm.bic(X), None, f)


def compare_algorithms(km_result: ClusterResult, gmm_result: ClusterResult,
                        km_stability: pd.DataFrame, gmm_stability: pd.DataFrame) -> ComparisonTable:
    """Choose between K-means and GMM using stability + interpretability + F-stat."""
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(km_result.labels, gmm_result.labels)

    km_mean_jac = km_stability["mean_jaccard"].mean()
    gmm_mean_jac = gmm_stability["mean_jaccard"].mean()

    # Decision rule (stated before looking at results):
    # 1. If mean Jaccard differs by >0.05 → prefer the stabler one
    # 2. Else → prefer the one with higher F-stat on 2010 cost (cleaner clinical narrative)
    if abs(km_mean_jac - gmm_mean_jac) > 0.05:
        chosen = "kmeans" if km_mean_jac > gmm_mean_jac else "gmm"
        justification = (
            f"K-means mean Jaccard={km_mean_jac:.2f} vs GMM={gmm_mean_jac:.2f}. "
            f"Chose {'K-means' if chosen == 'kmeans' else 'GMM'} on stability grounds."
        )
    else:
        chosen = "kmeans" if km_result.cost_f_stat >= gmm_result.cost_f_stat else "gmm"
        justification = (
            f"Stability similar (K-means Jac={km_mean_jac:.2f}, GMM={gmm_mean_jac:.2f}). "
            f"Chose {'K-means' if chosen == 'kmeans' else 'GMM'} on F-stat "
            f"({km_result.cost_f_stat:.0f} vs {gmm_result.cost_f_stat:.0f})."
        )

    log.info("Algorithm choice: %s — %s (ARI=%.2f)", chosen, justification, ari)
    return ComparisonTable(km_result, gmm_result, ari, chosen, justification)
