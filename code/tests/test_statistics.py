"""Small reference problems for inference; independent of the LastFM results."""
import itertools
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp

from lastfm_stats import (T_stat, T_pairs, binomial_interval, combine_strata,
                         holm_adjust, jaccard_matrix, leading_eigs, make_strata,
                         perm_test, score_ratios, scoreplus_M, summarize_null)


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 0, 1, 1, 1, 2, 2])
        rng = np.random.default_rng(9)
        raw = rng.uniform(size=(7, 7))
        self.similarity = (raw + raw.T) / 2
        np.fill_diagonal(self.similarity, 0)

    def test_equal_and_pair_weights_against_explicit_pairs(self):
        same = [[] for _ in range(3)]
        different = {(i, j): [] for i in range(3) for j in range(i + 1, 3)}
        for i, j in itertools.combinations(range(7), 2):
            a, b = sorted([self.labels[i], self.labels[j]])
            if a == b:
                same[a].append(self.similarity[i, j])
            else:
                different[a, b].append(self.similarity[i, j])
        expected = np.mean([np.mean(v) for v in same]) - np.mean([np.mean(v) for v in different.values()])
        pair_expected = np.mean(sum(same, [])) - np.mean(sum(different.values(), []))
        self.assertAlmostEqual(T_stat(self.similarity, self.labels), expected, places=13)
        self.assertAlmostEqual(T_pairs(self.similarity, self.labels), pair_expected, places=13)

    def test_music_and_inverse_label_permutation_are_equivalent(self):
        permutation = np.array([2, 5, 6, 1, 4, 0, 3])
        inverse = np.argsort(permutation)
        for statistic in [T_stat, T_pairs]:
            a = statistic(self.similarity[np.ix_(permutation, permutation)], self.labels)
            b = statistic(self.similarity, self.labels[inverse])
            self.assertAlmostEqual(a, b, places=13)

    def test_permutations_preserve_stratum_community_counts(self):
        strata = self.labels.copy()
        result = perm_test(self.similarity, self.labels, strata, 30)
        self.assertEqual(result["p"], 1)
        self.assertEqual(result["pair_result"]["p"], 1)
        self.assertTrue(np.allclose(result["null"], result["T"]))

    def test_tied_covariates_do_not_split_by_node_id(self):
        degree = np.array([1] * 9 + [2] * 6 + [3] * 5)
        activity = np.array([5] * 5 + [10] * 15)
        codes = make_strata(degree, activity, q=5)
        for i, j in itertools.combinations(range(20), 2):
            if degree[i] == degree[j] and activity[i] == activity[j]:
                self.assertEqual(codes[i], codes[j])
        self.assertEqual(len(set(make_strata(np.ones(20), q=5))), 1)
        self.assertEqual(len(combine_strata(codes, activity)), 20)

    def test_zero_exceedances_have_nonzero_p_and_nonzero_upper_bound(self):
        result = summarize_null(1, np.zeros(999))
        self.assertEqual(result["p"], 0.001)
        self.assertEqual(result["mc_tail_ci_low"], 0)
        self.assertGreater(result["mc_tail_ci_high"], 0.003)
        self.assertEqual(summarize_null(0, np.zeros(9))["p"], 1)
        self.assertEqual(binomial_interval(9, 9)[1], 1)

    def test_holm_known_example_and_original_order(self):
        np.testing.assert_allclose(holm_adjust([0.04, 0.001, 0.03]), [0.06, 0.003, 0.06])

    def test_scoreplus_extra_vector_uses_paper_signed_ratio(self):
        self.assertEqual(scoreplus_M(np.array([1., .8, .75]), 2), 3)
        self.assertEqual(scoreplus_M(np.array([1., .8, -.75]), 2), 2)

    def test_binary_jaccard_including_empty_sets(self):
        x = sp.csr_matrix([[1, 0, 1], [1, 1, 0], [0, 0, 0]])
        np.testing.assert_allclose(jaccard_matrix(x), [[0, 1 / 3, 0], [1 / 3, 0, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            jaccard_matrix(sp.csr_matrix([[2, 0], [1, 0]]))

    def test_refined_perron_vector_on_connected_bipartite_graph(self):
        a = sp.diags([np.ones(11), np.ones(11)], [-1, 1], shape=(12, 12), format="csr")
        values, vectors = leading_eigs(a, 3)
        self.assertTrue(np.all(vectors[:, 0] > 0))
        np.testing.assert_allclose(a @ vectors[:, 0], values[0] * vectors[:, 0], atol=1e-9)
        self.assertTrue(np.isfinite(score_ratios(vectors, 3, truncate=False)).all())


if __name__ == "__main__":
    unittest.main()
