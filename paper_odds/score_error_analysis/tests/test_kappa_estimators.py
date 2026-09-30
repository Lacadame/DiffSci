"""Conservation, exact-exponential recovery, and localized-peak sensitivity."""

import unittest
import numpy as np

from paper_odds.score_error_analysis.kappa_estimators import (
    fit_cumulative, primitive_at, smooth_profile, smoothed_mode_response,
)
from paper_odds.score_error_analysis.score_error_modes import fit_exponential


class KappaEstimatorTests(unittest.TestCase):
    def test_primitive_integrates_linear_segments_and_endpoint(self):
        d = np.array([0., .001, .7, 2., 5.])
        x = np.array([0., .0005, .2, 1., 5.])
        np.testing.assert_allclose(primitive_at(d, 2.-3*d, x), 2*x-1.5*x*x, atol=1e-14)

    def test_smoothing_conserves_signed_area_of_narrow_peak(self):
        d = np.unique(np.r_[0., np.geomspace(1e-7, 10., 600), np.linspace(0., 10., 200)])
        y = 20*np.exp(-4000*d)-.1*np.exp(-d)
        smoothed = smooth_profile(d, y, .05)
        self.assertAlmostEqual(smoothed['original_area'], smoothed['smoothed_area'], places=13)
        self.assertTrue(np.isfinite(smoothed['profile']).all())
        self.assertLess(smoothed['profile'].min(), 0.)
        # Endpoint-conservative smoothing preserves the shape ODE response exactly.
        actual = smoothed_mode_response(smoothed, [0.], mode='shape')[0]
        self.assertAlmostEqual(actual, np.trapz(y, d), places=13)

    def test_smoothing_preserves_constant_and_its_mean_response(self):
        d = np.array([0., .001, .07, .8, 4., 8.])
        smoothed = smooth_profile(d, np.full_like(d, -.2), .05)
        np.testing.assert_allclose(smoothed['profile'], -.2, atol=1e-12)
        gamma = np.array([0., 1., 5.])
        np.testing.assert_allclose(smoothed_mode_response(smoothed, gamma, mode='mean'),
                                   -.2*(1-np.exp(-(1+gamma)/2*8)), atol=1e-12)

    def test_cumulative_recovers_signed_exponential_for_both_modes(self):
        d = np.linspace(0, 8, 8001)
        for mode in ('shape', 'mean'):
            for amplitude, kappa in [(-.1, .7), (.03, -.2), (.2, 0.)]:
                fitted = fit_cumulative(d, amplitude*np.exp(-kappa*d), mode=mode)
                self.assertAlmostEqual(fitted['kappa'], kappa, places=5)
                self.assertAlmostEqual(fitted['amplitude'], amplitude, places=5)
                self.assertLess(fitted['relative_rmse'], 1e-6)

    def test_cumulative_does_not_chase_small_area_high_energy_peak(self):
        d = np.unique(np.r_[0., np.geomspace(1e-7, 10., 600), np.linspace(0., 10., 400)])
        y = .03*np.exp(-d)+5*np.exp(-4000*d)
        pointwise = fit_exponential(d, y)
        cumulative = fit_cumulative(d, y, mode='shape')
        self.assertGreater(pointwise['kappa'], 1000.)
        self.assertLess(abs(cumulative['kappa']-1.), .3)
        self.assertLess(cumulative['relative_rmse'], .03)

    def test_unresolved_localization_and_zero_profile_are_flagged(self):
        d = np.unique(np.r_[0., np.geomspace(1e-7, 10., 600), np.linspace(0., 10., 400)])
        fit = fit_cumulative(d, np.exp(-4000*d), mode='shape')
        self.assertTrue(fit['upper_unresolved'])
        zero = fit_cumulative(d, np.zeros_like(d), mode='mean')
        self.assertTrue(np.isnan(zero['kappa']))
        self.assertEqual(zero['amplitude'], 0.)

    def test_sign_cancellation_is_not_replaced_by_absolute_profile(self):
        d = np.linspace(0, 10, 1001)
        y = np.cos(d)
        signed = fit_cumulative(d, y, mode='shape')
        magnitude = fit_cumulative(d, abs(y), mode='shape')
        self.assertLess(signed['cancellation_ratio'], .1)
        self.assertAlmostEqual(magnitude['cancellation_ratio'], 1.)
        self.assertGreater(abs(signed['kappa']-magnitude['kappa']), .1)


if __name__ == '__main__':
    unittest.main()
