"""Independent small reference checks for the added statistical validation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp

import lastfm_stats as stats
from statistical_validation import (batch_permutation_test, bernoulli_dcbm,
                                    common_connected_core, _music_profiles)


class ValidationTests(unittest.TestCase):
    def test_batched_statistics_and_draws_match_existing_record_permutations(self):
        rng = np.random.default_rng(18)
        x = sp.csr_matrix((rng.random((9, 12)) < .4).astype(float))
        similarity = stats.jaccard_matrix(x)
        labels = {'three':np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                  'two':np.array([0, 0, 1, 1, 1, 1, 0, 0, 1])}
        strata = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
        batch = batch_permutation_test(similarity, labels, strata, 29, seed=82)
        for name, lab in labels.items():
            reference = stats.perm_test(similarity, lab, strata, 29, seed=82)
            for a, b in [(batch[name], reference), (batch[name]['pair_result'], reference['pair_result'])]:
                self.assertAlmostEqual(a['T'], b['T'], places=13)
                np.testing.assert_allclose(a['null'], b['null'], atol=1e-13)
                self.assertEqual(a['p'], b['p'])

    def test_batched_singletons_follow_explicit_equal_community_definition(self):
        similarity = np.full((6, 6), .2)
        np.fill_diagonal(similarity, 0)
        labels = {'partition':np.array([0, 0, 0, 1, 1, 2])}
        result = batch_permutation_test(similarity, labels, np.zeros(6), 9)
        self.assertAlmostEqual(result['partition']['T'], 0., places=13)
        self.assertEqual(result['partition']['p'], 1.)

    def test_bernoulli_generator_probability_and_mean_constraints(self):
        adjacency, truth, info = bernoulli_dcbm(300, 3, 12., 1.2, 8., 236)
        self.assertAlmostEqual(info['expected_mean_degree'], 12., places=10)
        self.assertLessEqual(info['maximum_probability'], 1.)
        self.assertGreaterEqual(info['theta_min'], .35-1e-12)
        self.assertLessEqual(info['theta_max'], 2.5+1e-12)
        self.assertEqual((adjacency != adjacency.T).nnz, 0)
        self.assertTrue(np.all(adjacency.diagonal() == 0))
        np.testing.assert_array_equal(np.bincount(truth), [100, 100, 100])
        other, other_truth, _ = bernoulli_dcbm(300, 3, 12., 0., 8., 236)
        np.testing.assert_array_equal(truth, other_truth)

    def test_common_core_has_the_same_vertices_and_is_connected_in_each_graph(self):
        first = sp.csr_matrix([[0,1,0,0], [1,0,1,0], [0,1,0,1], [0,0,1,0]])
        second = sp.csr_matrix([[0,1,1,0], [1,0,0,0], [1,0,0,0], [0,0,0,0]])
        core = common_connected_core([first, second])
        np.testing.assert_array_equal(core, [0, 1, 2])
        for graph in [first, second]:
            self.assertEqual(sp.csgraph.connected_components(graph[core][:, core], directed=False)[0], 1)

    def test_exact_binned_music_null_does_not_use_community_labels(self):
        degree = np.tile(np.arange(1, 7), 6)
        first_truth = np.repeat(np.arange(3), 12)
        second_truth = np.roll(first_truth, 4)
        first, first_count = _music_profiles(degree, first_truth, 'exact_binned_null', np.random.default_rng(8))
        second, second_count = _music_profiles(degree, second_truth, 'exact_binned_null', np.random.default_rng(8))
        self.assertEqual((first != second).nnz, 0)
        np.testing.assert_array_equal(first_count, second_count)
        stress_a, _ = _music_profiles(degree, first_truth, 'smooth_degree_only_stress', np.random.default_rng(9))
        stress_b, _ = _music_profiles(degree, second_truth, 'smooth_degree_only_stress', np.random.default_rng(9))
        self.assertEqual((stress_a != stress_b).nnz, 0)


if __name__ == '__main__':
    unittest.main()
