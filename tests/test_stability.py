"""Tests for the stability validation module."""

import numpy as np
import pytest

from patientseg.stability import _jaccard, cluster_stability


class TestJaccard:
    def test_identical_sets(self):
        assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial_overlap(self):
        assert abs(_jaccard({1, 2, 3}, {2, 3, 4}) - 0.5) < 1e-9

    def test_empty_sets(self):
        assert _jaccard(set(), set()) == 1.0


class TestClusterStability:
    def _make_well_separated_data(self, n_per_cluster=200, n_features=5, seed=42):
        """Create clearly separated clusters so stability should be near 1.0."""
        rng = np.random.default_rng(seed)
        X = np.vstack([
            rng.normal(loc=[0] * n_features, scale=0.3, size=(n_per_cluster, n_features)),
            rng.normal(loc=[5] * n_features, scale=0.3, size=(n_per_cluster, n_features)),
            rng.normal(loc=[10] * n_features, scale=0.3, size=(n_per_cluster, n_features)),
        ])
        labels = np.repeat([0, 1, 2], n_per_cluster)
        return X, labels

    def test_stable_kmeans_clusters(self):
        X, ref_labels = self._make_well_separated_data()
        result = cluster_stability(X, ref_labels, algorithm="kmeans", k=3, n_iter=5)
        assert len(result) == 3
        assert (result["mean_jaccard"] >= 0.0).all()
        assert (result["mean_jaccard"] <= 1.0).all()
        # Well-separated clusters should be highly stable
        assert result["mean_jaccard"].mean() > 0.7

    def test_stable_gmm_clusters(self):
        X, ref_labels = self._make_well_separated_data()
        result = cluster_stability(X, ref_labels, algorithm="gmm", k=3, n_iter=5)
        assert len(result) == 3
        assert result["mean_jaccard"].mean() > 0.7

    def test_output_schema(self):
        X, ref_labels = self._make_well_separated_data(n_per_cluster=100)
        result = cluster_stability(X, ref_labels, algorithm="kmeans", k=3, n_iter=3)
        expected_cols = {"cluster", "mean_jaccard", "std_jaccard", "n_stable_iters",
                         "n_iters", "stability_tier"}
        assert expected_cols.issubset(set(result.columns))

    def test_jaccard_in_unit_interval(self):
        """Jaccard scores must always be in [0, 1]."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 4))
        labels = np.repeat([0, 1, 2], 100)
        result = cluster_stability(X, labels, algorithm="kmeans", k=3, n_iter=4)
        assert (result["mean_jaccard"] >= 0.0).all()
        assert (result["mean_jaccard"] <= 1.0).all()

    def test_unknown_algorithm_raises(self):
        X = np.zeros((30, 2))
        labels = np.array([0] * 15 + [1] * 15)
        with pytest.raises(ValueError, match="Unknown algorithm"):
            cluster_stability(X, labels, algorithm="svm", k=2, n_iter=2)
