"""Checks of the score intervention, noise pairing, clock, and KL estimator."""

import unittest
import csv
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from paper_odds.score_error_analysis.score_error_modes import GaussianMixture1D
from paper_odds.score_error_analysis.residual_ablation import (
    AblationField, binned_kl, gaussian_moment_kl, histogram_counts, mixture_cdf,
    sample_paired, sampling_grid, shared_initial_states, target_partition, uniform_lookup,
)


class ResidualAblationTests(unittest.TestCase):
    def test_identical_arm_counts_have_zero_paired_effect_and_uncertainty(self):
        from paper_odds.score_error_analysis.run_residual_ablation import summarize_results
        config=dict(gammas=[0.,1.],lambdas=[0.,1.],seeds=[1,2,3,4],bins=[4],pseudocount=.5,particles=100)
        per_gamma=np.array([[[50,20,20,10],[45,25,20,10],[52,18,20,10],[48,22,20,10]],
                            [[60,20,10,10],[55,25,10,10],[62,18,10,10],[58,22,10,10]]])
        counts=np.repeat(per_gamma[:,None],2,axis=1)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'checkpoints').mkdir()
            np.savez(out/'exact_control.npz',counts_4=np.full((2,1,4,4),25))
            np.savez(out/'checkpoints/default3_epoch00.npz',counts_4=counts,
                     gaussian_profile_kl=np.array([.1,.2]),gaussian_moment_kl=np.array([[.1,.09],[.2,.19]]),
                     table_relative_rms_error_max=0.,table_nodes=33,fallback_points=0)
            summarize_results([dict(run='default3',epoch=0)],config,out)
            with (out/'paired_effects.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4)
            for row in rows:
                for key in ('nonlinear_effect_log10_ratio','nonlinear_effect_mc_se','disagreement_reduction','disagreement_reduction_mc_se'):
                    self.assertEqual(float(row[key]),0.)

    def test_brownian_bridge_preserves_the_coarse_increment_and_fine_marginals(self):
        from paper_odds.score_error_analysis.validate_residual_ablation import split_brownian_increment
        rng=np.random.default_rng(127)
        coarse=rng.normal(size=100000)
        first,second=split_brownian_increment(coarse,rng.normal(size=100000),.3,.7)
        np.testing.assert_allclose(first+second,coarse,atol=1e-15)
        self.assertAlmostEqual(first.var(),.3,delta=.006)
        self.assertAlmostEqual(second.var(),.7,delta=.012)
        self.assertAlmostEqual(np.cov(first,second)[0,1],0.,delta=.006)

    def test_lambda_endpoints_reproduce_the_intended_scores(self):
        mixture = GaussianMixture1D()
        clock = sampling_grid(mixture, steps=10)
        b = np.linspace(.1,.3,11)
        C = np.linspace(-.2,.4,11)
        model = lambda x,sigma: mixture.score(x,sigma)+.3+.1*np.sin(x)
        field = AblationField(mixture,clock,b,C,model=model)
        z = np.array([[-3.,-.2,0.,2.]])
        for index in (0,4,10):
            sv=clock['sqrt_variance'][index];x=mixture.mean+sv*z
            exact=sv*mixture.score(x,clock['sigma'][index])
            affine=exact+sv*b[index]+clock['variance'][index]*C[index]*z
            learned=sv*model(x,clock['sigma'][index])
            np.testing.assert_allclose(field.score(z,index,None),exact,atol=1e-13)
            np.testing.assert_allclose(field.score(z,index,0),affine,atol=1e-13)
            np.testing.assert_allclose(field.score(z,index,1),learned,atol=1e-13)
            np.testing.assert_allclose(field.score(z,index,.5),(affine+learned)/2,atol=1e-13)

    def test_identical_arms_are_identical_under_shared_noise(self):
        mixture=GaussianMixture1D()
        clock=sampling_grid(mixture,steps=80)
        model=lambda x,sigma: mixture.score(x,sigma)
        field=AblationField(mixture,clock,np.zeros(81),np.zeros(81),model=model)
        seeds=[33,71]
        initial=shared_initial_states(mixture,80,500,seeds)
        for integrator in ('heun', 'euler'):
            with self.subTest(integrator=integrator):
                result=sample_paired(field,[0.,1.,5.],[None,0.,1.],initial,seeds,integrator=integrator)
                np.testing.assert_array_equal(result[:,0],result[:,1])
                # Independent analytic evaluation may differ at machine precision.
                np.testing.assert_allclose(result[:,0],result[:,2],atol=1e-11,rtol=1e-11)
                second=sample_paired(field,[0.,1.,5.],[None,0.,1.],initial,seeds,integrator=integrator)
                np.testing.assert_array_equal(result,second)

    def test_euler_matches_discrete_gaussian_ou_solution_and_uses_one_score_per_step(self):
        from paper_odds.score_error_analysis.residual_ablation import brownian_generators
        mixture = GaussianMixture1D(means=(.3,), scales=(.7,), probabilities=(1.,))
        clock = sampling_grid(mixture, sigma_min=.1, sigma_max=2., steps=7)
        field = AblationField(mixture, clock, np.zeros(8), np.zeros(8))
        seeds = [19, 31]
        initial = shared_initial_states(mixture, 2., 100, seeds)
        gammas = [0., 1.]
        with patch.object(field, 'score', wraps=field.score) as score:
            actual = sample_paired(field, gammas, [None], initial, seeds, integrator='euler')
            self.assertEqual(score.call_count, 7)
        h = np.diff(clock['ell'])
        rngs = brownian_generators(seeds)
        increments = np.array([np.stack([rng.standard_normal(100) for rng in rngs])*np.sqrt(dt) for dt in h])
        for index, gamma in enumerate(gammas):
            # The exact discrete OU solution is a product of contractions and a
            # weighted sum of independent increments, not the continuous solution.
            contractions = 1-gamma*h/2
            tail_products = np.r_[np.cumprod(contractions[:0:-1])[::-1], 1.]
            final_z = contractions.prod()*initial + np.sqrt(gamma)*np.einsum('i,isp->sp', tail_products, increments)
            expected = mixture.mean+clock['sqrt_variance'][-1]*final_z
            np.testing.assert_allclose(actual[index, 0], expected, rtol=1e-13, atol=1e-13)

    def test_euler_gaussian_moments_match_discrete_constant_coefficient_solution(self):
        mixture = GaussianMixture1D()
        clock = sampling_grid(mixture, steps=500)
        V = clock['variance']
        a, u = .02, .04
        gammas = np.array([0., 1., 5.])
        with patch('paper_odds.score_error_analysis.residual_ablation.solve_ivp',
                   side_effect=AssertionError('Euler must not invoke the adaptive solver')):
            kl, r, v = gaussian_moment_kl(clock, u*np.sqrt(V[-1])/V, a/V, gammas, integrator='euler')
        h = np.diff(clock['ell'])[:, None]
        mean_product = np.prod(1-h*(1+gammas)*(1-a)/2, axis=0)
        expected_r = u/(1-a)*(1-mean_product)
        rate = -gammas+(1+gammas)*a
        variance_product = np.prod(1+h*rate, axis=0)
        expected_v = variance_product+gammas/rate*(variance_product-1)
        np.testing.assert_allclose(r, expected_r, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(v, expected_v, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(kl[:, 0], .5*(v-np.log(v)-1+r*r))

    def test_euler_refinement_uses_the_selected_method(self):
        from paper_odds.score_error_analysis.validate_residual_ablation import sample_coupled_refinement
        mixture = GaussianMixture1D(means=(0.,), scales=(1.,), probabilities=(1.,))
        coarse = sampling_grid(mixture, sigma_min=.1, sigma_max=2., steps=7)
        fine = sampling_grid(mixture, sigma_min=.1, sigma_max=2., steps=14)
        a = .1
        field = AblationField(mixture, fine, np.zeros(15), a/fine['variance'])
        seeds = [7, 13]
        initial = shared_initial_states(mixture, 2., 100, seeds)
        with patch.object(field, 'score', wraps=field.score) as score:
            samples = sample_coupled_refinement(field, [0.], [0.], initial, seeds, coarse, integrator='euler')
            self.assertEqual(score.call_count, 14)
        expected_z = initial*np.prod(1+a*np.diff(fine['ell'])/2)
        np.testing.assert_allclose(samples[0, 0], fine['sqrt_variance'][-1]*expected_z, atol=1e-13, rtol=1e-13)

    def test_unknown_integrators_fail_instead_of_silently_using_another_method(self):
        mixture = GaussianMixture1D()
        clock = sampling_grid(mixture, steps=4)
        field = AblationField(mixture, clock, np.zeros(5), np.zeros(5))
        initial = shared_initial_states(mixture, 80, 100, [3, 7])
        with self.assertRaisesRegex(ValueError, 'Integrator'):
            sample_paired(field, [0.], [None], initial, [3, 7], integrator='typo')
        with self.assertRaisesRegex(ValueError, 'Moment integrator'):
            gaussian_moment_kl(clock, np.zeros(5), np.zeros(5), [0.], integrator='typo')

    def test_exact_gaussian_probability_flow_preserves_standardized_particles(self):
        mixture=GaussianMixture1D(means=(.3,),scales=(.7,),probabilities=(1.,))
        clock=sampling_grid(mixture,steps=100)
        field=AblationField(mixture,clock,np.zeros(101),np.zeros(101))
        initial=shared_initial_states(mixture,80,500,[3,7])
        samples=sample_paired(field,[0.],[None],initial,[3,7])
        expected=mixture.mean+clock['sqrt_variance'][-1]*initial
        np.testing.assert_allclose(samples[0,0],expected,atol=1e-13)

    def test_exact_gaussian_stochastic_marginal(self):
        mixture=GaussianMixture1D(means=(0.,),scales=(.7,),probabilities=(1.,))
        clock=sampling_grid(mixture,steps=300)
        field=AblationField(mixture,clock,np.zeros(301),np.zeros(301))
        initial=shared_initial_states(mixture,80,12000,[9,19])
        samples=sample_paired(field,[1.,5.],[None],initial,[9,19])
        self.assertLess(abs(samples.mean()),.015)
        np.testing.assert_allclose(samples.var(axis=(-1,-2))[:,0],clock['variance'][-1],rtol=.035)

    def test_gaussian_moments_match_closed_form_for_constant_normalized_error(self):
        mixture=GaussianMixture1D()
        clock=sampling_grid(mixture,steps=60)
        V=clock['variance'];a=.02;u=.04
        gammas=np.array([0.,1.,5.])
        kl,r,v=gaussian_moment_kl(clock,np.full(len(V),u)*np.sqrt(V[-1])/V,np.full(len(V),a)/V,gammas)
        H=clock['ell'][-1]
        expected_r=u/(1-a)*(1-np.exp(-(1+gammas)*(1-a)/2*H))
        rate=-gammas+(1+gammas)*a
        expected_v=np.exp(rate*H)+gammas*np.expm1(rate*H)/rate
        np.testing.assert_allclose(r,expected_r,rtol=1e-7)
        np.testing.assert_allclose(v,expected_v,rtol=1e-7)
        np.testing.assert_allclose(kl[:,0],.5*(v-np.log(v)-1+r*r))

    def test_exact_target_partition_and_tail_accounting(self):
        mixture=GaussianMixture1D()
        edges=target_partition(mixture,.002,32)
        np.testing.assert_allclose(np.diff(mixture_cdf(edges,mixture,.002)),1/32,atol=1e-12)
        counts=histogram_counts(np.array([[-1e100,0.,1e100]]),edges)
        self.assertEqual(counts.sum(),3)
        self.assertEqual(counts[0,0],1)
        self.assertEqual(counts[0,-1],1)
        np.testing.assert_allclose(binned_kl(np.ones(32)*10),[0.,0.],atol=1e-14)
        self.assertTrue(np.isfinite(binned_kl(counts)).all())

    def test_lookup_and_direct_tail_fallback(self):
        mixture=GaussianMixture1D(means=(0.,),scales=(1.,),probabilities=(1.,))
        clock=sampling_grid(mixture,steps=4)
        grid=np.linspace(-2,2,33)
        table=np.array([-grid+clock['sqrt_variance'][i]*.2 for i in range(5)])
        model=lambda x,sigma: -x/(1+sigma*sigma)+.2
        field=AblationField(mixture,clock,np.zeros(5),np.zeros(5),table,model,2.)
        z=np.array([-8.,-.5,0.,1.,9.]);i=2;sv=clock['sqrt_variance'][i]
        np.testing.assert_allclose(field.score(z,i,1),sv*model(sv*z,clock['sigma'][i]))
        self.assertEqual(field.fallback_points,2)
        with self.assertRaises(ValueError):
            uniform_lookup(np.array([3.]),table[0],-2,2)


if __name__=='__main__':
    unittest.main()
