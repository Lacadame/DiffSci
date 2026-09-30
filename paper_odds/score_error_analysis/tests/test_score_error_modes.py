"""Numerical checks for the score-error decomposition (unittest, CPU only)."""

import ast
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad

from paper_odds.score_error_analysis.score_error_modes import (
    GaussianMixture1D, affine_projection, exponential_response, fit_exponential,
    normalized_profiles, profile_response,
)


class ProjectionTests(unittest.TestCase):
    def test_weighted_multidimensional_affine_recovery(self):
        rng = np.random.default_rng(101)
        x = rng.normal(size=(500, 2)) @ np.array([[2., 1.], [0., 0.4]]) + [3., -4.]
        w = rng.uniform(0, 1, len(x))
        center, b = np.array([2., -1.]), np.array([0.2, -0.5])
        C = np.array([[0.3, -0.7], [0.9, 0.4]])
        error = b + (x-center) @ C.T
        p = affine_projection(x, error, w, center)
        np.testing.assert_allclose(p['b'], b, atol=1e-13)
        np.testing.assert_allclose(p['C'], C, atol=1e-13)
        np.testing.assert_allclose(p['C_isotropic']+p['C_symmetric_traceless']+p['C_skew'], C)
        self.assertLess(p['residual_energy'], 1e-26)

    def test_nonlinear_hermite_residual_and_energy_identity(self):
        mixture = GaussianMixture1D(means=(0.,), scales=(1.,), probabilities=(1.,))
        x, w = mixture.quadrature(np.array([0.]), 32)
        x = x[0, :, None]
        error = 0.3 + 0.7*x + 0.2*(x*x-1)
        p = affine_projection(x, error, w)
        np.testing.assert_allclose(p['b'], [0.3], atol=1e-14)
        np.testing.assert_allclose(p['C'], [[0.7]], atol=1e-14)
        np.testing.assert_allclose([p['mean_energy'], p['affine_energy'], p['residual_energy']], [0.09, 0.49, 0.08], atol=1e-14)
        self.assertAlmostEqual(p['total_energy'], 0.66)
        self.assertLess(p['residual_mean_norm'], 1e-14)
        self.assertLess(p['residual_cross_norm'], 1e-14)

    def test_mixture_moments_and_stein_identities(self):
        mixture = GaussianMixture1D()
        sigma = np.array([0.002, 0.1, 1., 80.])
        x, w = mixture.quadrature(sigma, 512)
        np.testing.assert_allclose(x@w, mixture.mean, atol=1e-13)
        np.testing.assert_allclose(((x-mixture.mean)**2)@w, mixture.variance+sigma**2, rtol=1e-13)
        score = mixture.score(x, sigma[:, None])
        np.testing.assert_allclose(score@w, 0., atol=5e-7)
        np.testing.assert_allclose(((x-mixture.mean)*score)@w, -1., atol=5e-7)

    def test_rank_deficiency_is_not_silently_fitted(self):
        x = np.ones((10, 2))
        with self.assertRaises(ValueError):
            affine_projection(x, x)


class PhaseTests(unittest.TestCase):
    def test_exact_gaussian_mapping(self):
        V = np.array([1., 2., 8.])
        vref = 0.5
        log_alpha = np.array([0.02, -0.1, 0.3])
        u = np.array([0.01, 0.02, -0.01])
        C = (1-np.exp(-log_alpha))/V
        b = u*np.sqrt(vref)*np.exp(-log_alpha)/V
        out = normalized_profiles(b, C, V, vref)
        np.testing.assert_allclose(out['log_alpha'], log_alpha)
        np.testing.assert_allclose(out['u_exact'], u)
        invalid = normalized_profiles([0.1], [2.], [1.], 1.)
        self.assertFalse(invalid['exact_mapping_valid'][0])
        self.assertTrue(np.isnan(invalid['log_alpha'][0]))

    def test_signed_exponential_fit_and_zero_mode(self):
        d = np.unique(np.r_[np.linspace(0, 10, 181), np.linspace(0, .1, 301)])
        for amplitude, kappa in [(0.04, 1.3), (-0.08, -0.2), (0.1, 0.)]:
            f = fit_exponential(d, amplitude*np.exp(-kappa*d))
            self.assertAlmostEqual(f['amplitude'], amplitude, places=7)
            self.assertAlmostEqual(f['kappa'], kappa, places=6)
            self.assertLess(f['relative_rmse'], 1e-7)
        f = fit_exponential(d, np.zeros_like(d))
        self.assertTrue(np.isnan(f['kappa']))
        self.assertEqual(f['amplitude'], 0)

    def test_narrow_low_noise_profile_is_not_clipped_to_old_phase_axes(self):
        d = np.unique(np.r_[0., np.geomspace(1e-7, 10., 500), np.linspace(0., 10., 200)])
        f = fit_exponential(d, -.2*np.exp(-3500*d))
        self.assertAlmostEqual(f['kappa']/3500, 1., places=6)
        self.assertAlmostEqual(f['amplitude'], -.2, places=7)
        self.assertFalse(f['at_bound'])

    def test_signed_profile_cancellation_and_independent_kernels(self):
        d = np.linspace(0, 8., 20001)
        a, u = .03*np.cos(d), .05*np.sin(2*d)
        gammas = np.array([0., 1., 5.])
        r = profile_response(d, a, u, gammas)
        for i, g in enumerate(gammas):
            shape = quad(lambda t: (1+g)*np.exp(-g*t)*.03*np.cos(t), 0, 8)[0]
            mean = quad(lambda t: (1+g)/2*np.exp(-(1+g)/2*t)*.05*np.sin(2*t), 0, 8)[0]
            np.testing.assert_allclose([r['shape_response'][i], r['mean_response'][i]], [shape, mean], atol=3e-8)
            self.assertAlmostEqual(r['kl'][i], shape**2/4+mean**2/2, places=8)

    def test_phase_formulas_and_domain(self):
        gammas = np.array([0., .2, 1., 5.])
        kl = exponential_response(.6, -.2, .03, .02, gammas)
        expected = .03**2/4*((1+gammas)/(.6+gammas))**2 + .02**2/2*((1+gammas)/(.6+gammas))**2
        np.testing.assert_allclose(kl, expected)
        self.assertTrue(np.isnan(exponential_response(-.1, .2, 1., 1., gammas)).all())
        self.assertTrue(np.isfinite(exponential_response(-.1, -.5, 1., 1., gammas, 7.)).all())
        # Zero shape amplitude must leave a valid pure mean response.
        self.assertTrue(np.isfinite(exponential_response(np.nan, .1, 0., .02, gammas)).all())


class InferenceTests(unittest.TestCase):
    def test_matches_original_training_architecture_and_preconditioning(self):
        import torch
        from torch import nn
        from paper_odds.score_error_analysis.checkpoint_score import CheckpointScore
        root = next(p for p in Path(__file__).resolve().parents if (p/'diffsci').is_dir())
        paths = list((root/'savedmodels/production').glob('*-bs=16-*-nlm=default3/checkpoints/*epoch=00-*.ckpt'))
        if not paths:
            self.skipTest('Local model checkpoint unavailable')
        # Execute only the original two architecture definitions, not the script imports/main.
        tree = ast.parse((root/'stochasticity_paper/scripts/test-time_profile-correlation.py').read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('SinusoidalEmbedding', 'Improved1DMLP')]
        namespace = dict(torch=torch, nn=nn, math=math)
        exec(compile(ast.Module(body=classes, type_ignores=[]), '<original architecture>', 'exec'), namespace)
        original = namespace['Improved1DMLP'](residual=False).double()
        loaded = CheckpointScore(paths[0])
        original.load_state_dict(loaded.model.state_dict())
        x = torch.tensor([[-1.], [-.4], [.1], [.3], [80.]], dtype=torch.float64)
        sigma = torch.tensor([.002, .02, .2, 2., 80.], dtype=torch.float64)
        with torch.no_grad():
            raw = original(x/torch.sqrt(sigma**2+.25)[:, None], .5*torch.log(sigma))
            denoiser = (.25/(sigma**2+.25))[:, None]*x + (.5*sigma/torch.sqrt(sigma**2+.25))[:, None]*raw
            expected = ((denoiser-x)/sigma[:, None]**2)[:, 0].numpy()
        np.testing.assert_allclose(loaded(x[:, 0].numpy(), sigma.numpy()), expected, rtol=1e-9, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
