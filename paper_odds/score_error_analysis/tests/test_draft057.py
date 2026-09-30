"""Analytic and simulation checks for the draft's changed scientific calculations."""
import unittest
import numpy as np
from scipy.integrate import solve_ivp

from ..draft057 import (responses, two_amplitude_odds, shape_gaussian_odds,
                        gaussian_joint_odds, bounded_change, bounded_from_log10,
                        affine_moments, fit_log_noise_gp, gp_response_distribution)


class Draft057Tests(unittest.TestCase):
    def test_constant_profile_response(self):
        d = np.linspace(0, 8, 20001)
        g = np.array([0., .2, 1., 5.])
        result = responses(d, np.full_like(d, .03), np.full_like(d, -.1), g)
        expected_L = np.r_[8*.03, .03*(1+g[1:])*(-np.expm1(-g[1:]*8))/g[1:]]
        expected_M = -.1*(-np.expm1(-(1+g)/2*8))
        np.testing.assert_allclose(result['L'], expected_L, rtol=4e-7)
        np.testing.assert_allclose(result['M'], expected_M, rtol=2e-7)

    def test_two_amplitudes_both_mixed_cases(self):
        self.assertAlmostEqual(float(two_amplitude_odds(-2, 1)), .5)
        self.assertAlmostEqual(float(two_amplitude_odds(2, -1)), .5)
        np.testing.assert_equal(two_amplitude_odds([-1, 1, 0, 0], [-1, 1, 0, -1]), [1, 0, 0, 1])
        rng = np.random.default_rng(12)
        z = rng.standard_normal((300000, 2))
        for da, du in [(-1, 3), (2, -.3)]:
            observed = np.mean(da*z[:, 0]**2/4+du*z[:, 1]**2/2 < 0)
            self.assertAlmostEqual(float(two_amplitude_odds(da, du)), observed, delta=.003)

    def test_orthant_formula_centered_noncentered_singular(self):
        self.assertAlmostEqual(shape_gaussian_odds([0, 0], np.diag([4, 1])), 2/np.pi*np.arctan(2))
        self.assertEqual(shape_gaussian_odds([0, 0], [[1, 1], [1, 1]]), 0)
        self.assertEqual(shape_gaussian_odds([0, 0], [[1, .5], [.5, .25]]), 1)
        rng = np.random.default_rng(123)
        mean, cov = np.array([.7, -.2]), np.array([[1., .25], [.25, .4]])
        z = rng.multivariate_normal(mean, cov, 400000)
        self.assertAlmostEqual(shape_gaussian_odds(mean, cov), np.mean(z[:, 1]**2 < z[:, 0]**2), delta=.003)

    def test_joint_probability_matches_two_amplitude_law(self):
        # [L0,Lg]=[2,1] Za and [M0,Mg]=[1,2] Zu.
        f = np.array([[2., 0], [1, 0], [0, 1], [0, 2]])
        answer = gaussian_joint_odds(np.zeros(4), f@f.T, power=15)
        self.assertAlmostEqual(answer['probability'], float(two_amplitude_odds(-3, 3)), delta=.002)
        self.assertLess(answer['numerical_se'], .002)

    def test_bounded_statistic(self):
        h = np.array([.1, 1, 10])
        np.testing.assert_allclose(bounded_change(h, 1), bounded_from_log10(np.log10(h)))
        self.assertTrue(np.isnan(bounded_change(0, 0)))
        np.testing.assert_equal(bounded_change([0, 1], [1, 0]), [-1, 1])

    def test_moments_static_and_sign_changing_a_above_one(self):
        d = np.linspace(0, 2, 101)
        g = np.array([0., 1., 5.])
        a, u = .05, -.03
        kl, r, v = affine_moments(d, d*0+a, d*0+u, g)
        k = -g+(1+g)*a
        expected_v = np.exp(k*2)+g*np.expm1(k*2)/k
        expected_r = u/(1-a)*(-np.expm1(-(1+g)/2*(1-a)*2))
        np.testing.assert_allclose(v, expected_v, rtol=1e-12)
        np.testing.assert_allclose(r, expected_r, rtol=1e-12)
        a, u = 1.2*np.cos(4*d), .05*np.sin(2*d)
        _, r, v = affine_moments(d, a, u, [1], refinement=16)
        def rhs(ell, state):
            at, ut = np.interp(2-ell, d, a), np.interp(2-ell, d, u)
            return [-(1-at)*state[0]+ut, (-1+2*at)*state[1]+1]
        reference = solve_ivp(rhs, [0, 2], [0, 1], rtol=1e-10, atol=1e-12, max_step=.002).y[:, -1]
        np.testing.assert_allclose([r[0], v[0]], reference, rtol=4e-6)

    def test_ou_length_recovery_and_projected_covariance(self):
        x = np.linspace(-3, 3, 257)
        rng = np.random.default_rng(3)
        length = .65
        rho = np.exp(-(x[1]-x[0])/length)
        z = rng.standard_normal((1500, len(x)))
        for i in range(1, len(x)):
            z[:, i] = rho*z[:, i-1]+np.sqrt(1-rho*rho)*z[:, i]
        result = fit_log_noise_gp(np.exp(x), z)
        self.assertAlmostEqual(result['length'], length, delta=.06)
        mean, cov = gp_response_distribution(x-x[0], np.exp(x), x*0, x*0+1, length, [0, .2, 1])
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-12)
        np.testing.assert_equal(mean, 0)


if __name__ == '__main__':
    unittest.main()
