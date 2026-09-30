"""Identity-line scoring, unbiased comparisons, and gamma aggregation checks."""

import unittest
import numpy as np

from paper_odds.score_error_analysis.prediction_agreement import (
    aggregate, evaluate, score_predictions,
)


class PredictionAgreementTests(unittest.TestCase):
    def test_perfect_prediction_has_perfect_agreement(self):
        actual = [-.4, -.1, .1, .4]
        result = score_predictions(actual, actual)
        for scale in ('log10', 'ratio'):
            self.assertEqual(result[f'mse_{scale}'], 0.)
            self.assertEqual(result[f'mae_{scale}'], 0.)
            self.assertEqual(result[f'concordance_{scale}'], 1.)
            self.assertEqual(result[f'identity_r2_{scale}'], 1.)
            self.assertEqual(result[f'skill_vs_no_change_{scale}'], 1.)

    def test_perfect_correlation_does_not_hide_bias(self):
        actual = np.array([-.4, -.1, .1, .4])
        result = score_predictions(actual+1., actual)
        self.assertAlmostEqual(result['spearman'], 1.)
        self.assertAlmostEqual(result['pearson_log10'], 1.)
        self.assertAlmostEqual(result['rmse_log10'], 1.)
        self.assertAlmostEqual(result['bias_log10'], 1.)
        self.assertLess(result['identity_r2_log10'], 0.)
        self.assertLess(result['skill_vs_no_change_log10'], 0.)
        self.assertLess(result['concordance_log10'], .2)

    def test_no_change_baseline_has_zero_skill_on_both_scales(self):
        result = score_predictions([0, 0, 0], [-.2, .1, .3])
        self.assertEqual(result['skill_vs_no_change_log10'], 0.)
        self.assertEqual(result['skill_vs_no_change_ratio'], 0.)
        self.assertTrue(np.isnan(result['spearman']))

    def test_raw_ratio_least_squares_and_log_error_are_distinct(self):
        result = score_predictions(np.log10([1., 4., 4.]), np.log10([1., 2., 4.]))
        self.assertAlmostEqual(result['mse_ratio'], 4/3)
        self.assertAlmostEqual(result['mae_ratio'], 2/3)
        self.assertAlmostEqual(result['mse_log10'], np.log10(2.)**2/3)
        opposite = score_predictions([-1., 1.], [0., 0.])
        self.assertEqual(opposite['rmse_log10'], 1.)
        self.assertAlmostEqual(opposite['mse_ratio'], (.9**2+9.**2)/2)

    def test_constant_perfect_and_empty_inputs_are_explicit(self):
        result = score_predictions([0., 0.], [0., 0.])
        self.assertEqual(result['concordance_log10'], 1.)
        self.assertTrue(np.isnan(result['identity_r2_log10']))
        self.assertTrue(np.isnan(result['skill_vs_no_change_log10']))
        empty = score_predictions([], [])
        self.assertEqual(empty['n'], 0)
        self.assertTrue(np.isnan(empty['rmse_log10']))
        with self.assertRaises(ValueError):
            score_predictions([0., np.nan], [0., 1.])

    def test_aggregation_weights_gammas_equally_not_by_sample_count(self):
        rows = []
        for gamma, pred, actual in [(.2, [0.]*9, [1.]*9), (1., [3.], [1.])]:
            rows.append(dict(dataset='test', bins=64, kl_direction='q_p', subset='all', run='all',
                             method='test', gamma=gamma, **score_predictions(pred, actual)))
        result = aggregate(rows)[0]
        self.assertEqual(result['n_pairs'], 10)
        self.assertAlmostEqual(result['mse_log10'], (1+4)/2)
        self.assertAlmostEqual(result['rmse_log10'], np.sqrt(2.5))
        self.assertAlmostEqual(result['skill_vs_no_change_log10'], 1-2.5)

    def test_evaluation_uses_common_support_for_every_method(self):
        from paper_odds.score_error_analysis.prediction_agreement import LABELS
        records = []
        for epoch, common in enumerate((True, True, False)):
            records.append(dict(dataset='controlled_full', gamma=1., baseline_gamma=0., bins=64,
                                kl_direction='q_p', run='a', epoch=epoch, measured_log10_ratio=.2,
                                predictions={method: .1 for method in LABELS},
                                common_finite=common, above_control_floor=epoch == 0))
        results = evaluate(records)
        for r in results:
            self.assertEqual(r['n'], 2 if r['subset'] == 'all' else 1)
            self.assertEqual(r['excluded_common_nonfinite'], 1 if r['subset'] == 'all' else 0)


if __name__ == '__main__':
    unittest.main()
