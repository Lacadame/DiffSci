"""Known slopes, sign handling, near-zero masking, and hybrid-grid weighting."""

import unittest
import numpy as np

from paper_odds.score_error_analysis.loglog_kappa import fit_loglog
from paper_odds.score_error_analysis.score_error_modes import integration_weights


class LogLogKappaTests(unittest.TestCase):
    def test_known_power_laws_and_log_base_invariance(self):
        d = np.unique(np.r_[np.linspace(0, .1, 51), np.linspace(.1, 8, 100)])
        for amplitude, kappa in [(.03, .7), (-.05, -.2), (.2, 0.)]:
            y = amplitude*np.exp(-kappa*d)
            for weighting in ('points', 'log_variance'):
                fit = fit_loglog(d, y, weighting=weighting)
                self.assertAlmostEqual(fit['kappa'], kappa, places=12)
                self.assertAlmostEqual(fit['amplitude'], abs(amplitude), places=12)
                self.assertLess(fit['log_rmse'], 1e-12)
            slope = np.polyfit(np.log10(np.exp(d)), np.log10(abs(y)), 1)[0]
            self.assertAlmostEqual(-slope, kappa, places=12)

    def test_sign_changes_are_retained_and_flagged_as_envelope(self):
        d = np.linspace(0, 8, 201)
        positive = .1*np.exp(-.8*d)
        signed = positive*np.where(np.sin(3*d) >= 0, 1, -1)
        plain, mixed = fit_loglog(d, positive), fit_loglog(d, signed)
        self.assertTrue(mixed['mixed_sign'])
        self.assertGreater(mixed['sign_changes'], 3)
        self.assertEqual(mixed['retained_count'], len(d))
        self.assertEqual(plain['kappa'], mixed['kappa'])
        self.assertEqual(plain['amplitude'], mixed['amplitude'])

    def test_masked_nodes_keep_original_clock_weights(self):
        d = np.array([0., .01, .05, .1, .5, 1., 2., 4.])
        y = np.array([1., .8, 0., 1e-15, -.3, .15, .2, .08])
        mask = abs(y) > 1e-6
        w = integration_weights(d)[mask]
        X = np.column_stack([np.ones(mask.sum()), d[mask]])
        expected = np.linalg.lstsq(X*np.sqrt(w[:, None]), np.log(abs(y[mask]))*np.sqrt(w), rcond=None)[0]
        fit = fit_loglog(d, y)
        np.testing.assert_allclose([fit['log_amplitude'], fit['slope']], expected, atol=1e-13)
        self.assertEqual(fit['excluded_count'], 2)
        self.assertAlmostEqual(fit['retained_clock_fraction'], w.sum()/integration_weights(d).sum())

    def test_weighting_prevents_dense_terminal_nodes_from_domination(self):
        d = np.unique(np.r_[np.linspace(0, .1, 10001), np.linspace(.1, 10, 1001)])
        y = np.exp(.2+.5*d+.03*d*d)
        weighted, ordinary = fit_loglog(d, y), fit_loglog(d, y, weighting='points')
        # Uniform-clock OLS slope of d² over [0,L] is L; expected slope=.5+.03*10.
        self.assertAlmostEqual(weighted['slope'], .8, places=5)
        self.assertGreater(abs(ordinary['slope']-.8), .03)

    def test_large_true_exponent_is_not_clipped(self):
        d = np.linspace(0, .001, 101)
        fit = fit_loglog(d, -.02*np.exp(-4000*d))
        self.assertAlmostEqual(fit['kappa'], 4000., places=7)

    def test_zero_and_undersampled_profiles_are_undefined(self):
        d = np.linspace(0, 1, 5)
        self.assertFalse(fit_loglog(d, np.zeros_like(d))['valid'])
        self.assertFalse(fit_loglog(d, [1., 0., 0., 0., 1.])['valid'])
        with self.assertRaises(ValueError):
            fit_loglog(d, [1, 2, np.nan, 4, 5])
        with self.assertRaises(ValueError):
            fit_loglog(d, np.ones_like(d), relative_cutoff=1.)


if __name__ == '__main__':
    unittest.main()
