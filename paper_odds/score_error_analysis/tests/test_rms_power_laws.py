"""Recovery and model-family checks for checkpoint RMS power-law envelopes."""
import unittest
import numpy as np

from paper_odds.score_error_analysis.rms_power_laws import (
    evaluate_power_law, fit_piecewise_power_laws,
)


class PiecewisePowerTests(unittest.TestCase):
    def test_recovers_three_segments_with_unknown_breakpoints_on_irregular_grid(self):
        t = np.unique(np.r_[np.geomspace(.002, 80, 130), np.geomspace(.003, .1, 50)])
        knots, powers, amplitude = [.018, .9], [.6, 1.2, .1], 4.
        y = np.where(t <= knots[0], amplitude*(t/.002)**-powers[0],
                     np.where(t <= knots[1], amplitude*(knots[0]/.002)**-powers[0]
                              *(t/knots[0])**-powers[1],
                              amplitude*(knots[0]/.002)**-powers[0]
                              *(knots[1]/knots[0])**-powers[1]*(t/knots[1])**-powers[2]))
        fits = fit_piecewise_power_laws(t, y, seed=17, restarts=2)
        fit = fits[-1]
        np.testing.assert_allclose(fit['exponents'], powers, atol=2e-5)
        np.testing.assert_allclose(fit['breakpoints'], knots, rtol=1e-4)
        self.assertAlmostEqual(fit['amplitude'], amplitude, places=4)
        self.assertLess(fit['log10_RMSE'], 1e-6)
        self.assertTrue(np.all(np.diff([f['objective'] for f in fits]) <= 1e-12))
        for knot in fit['breakpoints']:
            values = evaluate_power_law(np.array([knot*(1-1e-8), knot, knot*(1+1e-8)]), fit)
            np.testing.assert_allclose(values, values[1], rtol=1e-7)
        self.assertTrue(np.all(np.diff(evaluate_power_law(t, fit)) <= 0))

    def test_magnitude_fit_recovers_decline_and_plateau(self):
        t = np.geomspace(.002, 30, 110)
        y = 2.1*(np.minimum(t, .3)/.002)**-.8
        fits = fit_piecewise_power_laws(t, y, max_segments=2, fit_space='magnitude',
                                       seed=9, restarts=2)
        fit = fits[-1]
        np.testing.assert_allclose(fit['exponents'], [.8, 0.], atol=2e-5)
        np.testing.assert_allclose(fit['breakpoints'], [.3], rtol=1e-4)
        self.assertLess(fit['relative_RMSE'], 1e-6)
        self.assertFalse(fit['rate_bound_hit'])

    def test_single_power_weighted_log_fit_and_constant_boundary(self):
        t = np.geomspace(.01, 10, 60)
        w = np.linspace(.1, 3, len(t)); w /= w.sum()
        y = 5*(t/t[0])**-.7
        fit = fit_piecewise_power_laws(t, y, w, max_segments=1)[0]
        np.testing.assert_allclose(fit['exponents'], [.7], atol=1e-12)
        self.assertAlmostEqual(fit['amplitude'], 5., places=12)
        rising = t**.5
        flat = fit_piecewise_power_laws(t, rising, w, max_segments=1)[0]
        self.assertEqual(flat['exponents'][0], 0.)
        self.assertAlmostEqual(flat['amplitude'], np.exp(w@np.log(rising)), places=12)

    def test_rejects_invalid_data_instead_of_silently_dropping_points(self):
        t = np.geomspace(.01, 10, 10)
        for bad in [np.full(10, np.nan), np.zeros(10), -np.ones(10)]:
            with self.assertRaises(ValueError):
                fit_piecewise_power_laws(t, bad)
        with self.assertRaises(ValueError):
            fit_piecewise_power_laws(t[::-1], np.ones(10))
        with self.assertRaises(ValueError):
            fit_piecewise_power_laws(t, np.ones(10), max_segments=4)


if __name__ == '__main__':
    unittest.main()
