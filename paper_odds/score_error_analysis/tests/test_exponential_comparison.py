"""Checks for baseline matching, correlation scales, and phase-color joins."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad

from paper_odds.score_error_analysis.compare_exponential_predictions import (
    agreement_metrics, exponential_log_ratio,
)
from paper_odds.score_error_analysis.plot_checkpoint_modes import phase_measurements


class ExponentialComparisonTests(unittest.TestCase):
    def test_exponential_prediction_uses_requested_baseline_and_signed_amplitudes(self):
        cp = dict(kappa_a=.3, kappa_m=-.2, epsilon_a=-.04, epsilon_m=.07, horizon=7.)
        def independent(g):
            shape = quad(lambda d: (1+g)*cp['epsilon_a']*np.exp(-(cp['kappa_a']+g)*d), 0, 7)[0]
            mean = quad(lambda d: (1+g)/2*cp['epsilon_m']*np.exp(-(cp['kappa_m']+(1+g)/2)*d), 0, 7)[0]
            return shape**2/4+mean**2/2
        for baseline in (0., .01):
            self.assertAlmostEqual(exponential_log_ratio(cp, 1., baseline),
                                   np.log10(independent(1.)/independent(baseline)), places=12)
            self.assertAlmostEqual(exponential_log_ratio(cp, baseline, baseline), 0.)
        self.assertNotEqual(exponential_log_ratio(cp, 1., 0.), exponential_log_ratio(cp, 1., .01))

    def test_infinite_domain_and_zero_mode(self):
        cp = dict(kappa_a=-.1, kappa_m=.1, epsilon_a=.04, epsilon_m=.07, horizon=7.)
        self.assertTrue(np.isfinite(exponential_log_ratio(cp, 1., 0.)))
        self.assertTrue(np.isnan(exponential_log_ratio(cp, 1., 0., infinite=True)))
        cp.update(epsilon_a=0., kappa_a=np.nan)
        self.assertTrue(np.isfinite(exponential_log_ratio(cp, 1., 0., infinite=True)))

    def test_correlations_mask_nonfinite_and_distinguish_scales(self):
        result = agreement_metrics([0, 1, 2, np.nan], [0, 2, 4, 8], ['a']*4)
        self.assertEqual(result['n'], 3)
        self.assertEqual(result['excluded_nonfinite'], 1)
        self.assertAlmostEqual(result['spearman'], 1.)
        self.assertAlmostEqual(result['pearson_log10'], 1.)
        self.assertLess(result['pearson_ratio'], 1.)
        self.assertEqual(result['median_absolute_log10_gap'], 1.)
        self.assertTrue(np.isnan(agreement_metrics([1, 1, 1], [1, 2, 3], ['a']*3)['spearman']))
        self.assertEqual(agreement_metrics([], [], [])['n'], 0)

    def test_within_run_centering_detects_reversed_within_run_association(self):
        result = agreement_metrics([0, 1, 2, 10, 11, 12], [2, 1, 0, 12, 11, 10], ['a']*3+['b']*3)
        self.assertGreater(result['pearson_log10'], .9)
        self.assertAlmostEqual(result['pearson_log10_within_run_centered'], -1.)

    def test_phase_colors_join_exact_gamma_full_arm_and_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp)
            (paper/'outputs/residual_ablation').mkdir(parents=True)
            path = paper/'outputs/residual_ablation/ablation_results.csv'
            rows = [dict(run='test', epoch=7, gamma=g, residual_lambda=lam, bins=b,
                         log10_ratio_ode_q_p=v, log10_ratio_ode_p_q=-v)
                    for g, lam, b, v in [(1., 1., 64, .3), (.2, 1., 64, .5), (1., 0., 64, .9), (1., 1., 32, .8)]]
            def write():
                with path.open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            write()
            measurements = phase_measurements(paper)
            self.assertEqual(len(measurements), 2)
            self.assertEqual(measurements['test', 7, 1.]['q_p'], .3)
            self.assertEqual(measurements['test', 7, .2]['p_q'], -.5)
            rows.append(rows[0]); write()
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                phase_measurements(paper)


if __name__ == '__main__':
    unittest.main()
