"""Scientifically relevant checks for paired time-dependent gamma sampling."""
import tempfile
import unittest
import numpy as np

from ..schedule_experiment import (common_clock, schedule_matrix, sample_schedules,
    split_profile, cell_response, cell_moments, prepare_cache)
from ..score_error_modes import GaussianMixture1D
from ..residual_ablation import (sampling_grid, AblationField, sample_paired, shared_initial_states)


class ScheduleTests(unittest.TestCase):
    def test_constant_schedule_is_existing_sampler(self):
        mixture=GaussianMixture1D()
        clock=sampling_grid(mixture,steps=60)
        n=len(clock['sigma'])
        field=AblationField(mixture,clock,np.ones(n)*.03,np.ones(n)*-.02)
        seeds=[170,271]
        initial=shared_initial_states(mixture,80.,100,seeds)
        gamma=np.array([0.,.2,1.,5.])
        actual=sample_schedules(field,np.broadcast_to(gamma[:,None],(4,n-1)),[None,0.],initial,seeds)
        expected=sample_paired(field,gamma,[None,0.],initial,seeds,integrator='euler')
        np.testing.assert_array_equal(actual,expected)

    def test_common_clock_contains_narrow_window_edges(self):
        mixture=GaussianMixture1D()
        spec=dict(kind='window',gamma=50.,lo=.40001,hi=.40002)
        clock=common_clock(mixture,.002,80.,50,[spec],max_step_contraction=.1)
        for edge in [spec['lo'],spec['hi']]:
            self.assertLess(np.min(np.abs(clock['sigma']-edge)),1e-12)
        gamma=schedule_matrix(clock,mixture,[spec])[0]
        budget=gamma@np.diff(clock['ell'])
        expected=50*np.log((mixture.variance+spec['hi']**2)/(mixture.variance+spec['lo']**2))
        self.assertAlmostEqual(budget,expected,places=12)
        self.assertTrue(np.all(np.diff(clock['ell'])>0))

    def test_constant_profile_response_and_moments(self):
        s=np.geomspace(.002,80,101);v0=.1219
        a,u=np.full_like(s,.02),np.full_like(s,-.03)
        for g in [0.,.2,1.,50.]:
            grid,aa,uu,gg=split_profile(s,a,u,v0,dict(kind='const',gamma=g))
            L,M,h=cell_response(grid,aa,uu,gg);D=grid[-1]
            expected_L=.02*D if g==0 else .02*(1+g)*(-np.expm1(-g*D))/g
            expected_M=-.03*(-np.expm1(-(1+g)*D/2))
            np.testing.assert_allclose([L,M],[expected_L,expected_M],rtol=1e-13)
            c=-g+(1+g)*.02
            v=np.exp(c*D)+g*np.expm1(c*D)/c
            r=-.03/.98*(-np.expm1(-(1+g)/2*.98*D))
            expected=[.5*(v-np.log(v)-1+r*r),.5*(np.log(v)+1/v-1+r*r/v)]
            np.testing.assert_allclose(cell_moments(grid,aa,uu,gg),expected,rtol=1e-10,atol=1e-13)

    def test_duplicate_schedules_and_arms_share_identical_noise(self):
        mix=GaussianMixture1D();clock=sampling_grid(mix,steps=70)
        field=AblationField(mix,clock,np.zeros(71),np.zeros(71))
        seeds=[170,271];init=shared_initial_states(mix,80.,100,seeds)
        specs=[dict(kind='window',gamma=1.,lo=.2,hi=1.)]*2
        out=sample_schedules(field,schedule_matrix(clock,mix,specs),[None,0.],init,seeds)
        np.testing.assert_array_equal(out[0],out[1])
        np.testing.assert_array_equal(out[:,0],out[:,1])

    def test_cache_changes_with_experiment_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1,h1=prepare_cache(tmp,dict(seed=1,schedules=[dict(kind='window',lo=0,hi=np.inf)]))
            p2,h2=prepare_cache(tmp,dict(seed=2,schedules=[dict(kind='window',lo=0,hi=np.inf)]))
            self.assertNotEqual(p1,p2);self.assertNotEqual(h1,h2)
            self.assertEqual(prepare_cache(tmp,dict(seed=1,schedules=[dict(kind='window',lo=0,hi=np.inf)])),(p1,h1))


if __name__=='__main__': unittest.main()
