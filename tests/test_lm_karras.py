import ast
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import diffsci.models

HELPERS = Path(__file__).resolve().parents[1] / 'notebooks/exploratory/bps/karras_integrator'
sys.path.insert(0, str(HELPERS))
from comparison_utils import LMKarrasIntegrator, gaussian_reverse_moments


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def matched_type():
    notebook = json.loads((HELPERS / '02-mixtures-learned_scores.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells']
                  if 'class GammaMatchedKarrasIntegrator' in ''.join(c['source']))
    classes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
               and n.name in ('GammaMatchedKarrasIntegrator', 'GammaMatchedLMKarrasIntegrator')]
    namespace = dict(diffsci=diffsci, np=np, KARRAS_CHURN_CAP=None,
                     LMKarrasIntegrator=LMKarrasIntegrator)
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<notebook>', 'exec'), namespace)
    return namespace['GammaMatchedLMKarrasIntegrator']


def discrete_gaussian(times, gamma):
    mean = .7
    variance = 1.8*(.25 + times[0]**2)
    covariance = 0.
    for t, next_t in zip(times[:-1], times[1:]):
        hat = t + gamma*(t-next_t)
        dt = next_t-hat
        a, b = hat/(.25+hat**2), next_t/(.25+next_t**2)
        amplification = 1 + .5*dt*(a+b*(1+dt*a))
        c = .5*amplification*np.sqrt(hat**2-t**2)
        mean = amplification*mean
        variance = amplification**2*variance + 2*c*c + 2*amplification*c*covariance
        covariance = c
    return mean, variance


def test_free_noise_variance_and_adjacent_covariance():
    method = LMKarrasIntegrator(s_schurn=.2, s_tmin=None, s_noise=1, churn_cap=None)
    scheduler = diffsci.models.EDMScheduler()
    torch.manual_seed(123)
    x = torch.zeros(100000, 1)
    increments = []
    variance_per_kick = 1.2**2-1
    for _ in range(25):
        next_x = method.step(x, torch.tensor(1.), torch.tensor(-.1),
                             lambda x, t: torch.zeros_like(x), scheduler.scheduler_fns, nsteps=1)
        increments.append(next_x-x)
        x = next_x
    assert float(increments[0].var()) == pytest.approx(variance_per_kick/2, rel=.02)
    assert float((increments[0]*increments[1]).mean()) == pytest.approx(variance_per_kick/4, rel=.025)
    assert float(x.var()) == pytest.approx(variance_per_kick*(25-.5), rel=.02)


@pytest.mark.parametrize('gamma', [0., 1., 4.])
def test_actual_matched_sampler_agrees_with_gaussian_moments(gamma):
    times = np.linspace(2**(1/7), .05**(1/7), 49)**7
    method = matched_type()(gamma)
    scheduler = diffsci.models.EDMScheduler().double()
    torch.manual_seed(321)
    x = .7 + np.sqrt(1.8*(.25+times[0]**2))*torch.randn(120000, 1, dtype=torch.float64)
    for t, next_t in zip(times[:-1], times[1:]):
        x = method.step(x, torch.tensor(t), torch.tensor(next_t-t),
                        lambda x, t: t*x/(.25+t*t), scheduler.scheduler_fns, nsteps=48)
    mean, variance = discrete_gaussian(times, gamma)
    assert float(x.mean()) == pytest.approx(mean, abs=6*np.sqrt(variance/len(x)))
    assert float(x.var()) == pytest.approx(variance, abs=6*variance*np.sqrt(2/(len(x)-1)))


@pytest.mark.parametrize('gamma', [1., 4.])
def test_gaussian_refinement_approaches_the_same_reverse_sde(gamma):
    errors = []
    for n in (100, 200, 400, 800, 1600):
        times = np.linspace(2**(1/7), .05**(1/7), n+1)**7
        mean, variance = discrete_gaussian(times, gamma)
        exact_mean, exact_variance = gaussian_reverse_moments(times, gamma, 'exact')
        errors.append(abs(mean-exact_mean[-1]) + abs(variance-exact_variance[-1]))
    assert errors[-1] < errors[0]/8
    assert errors[-1] < .005


def test_zero_gamma_matches_karras_and_memory_resets():
    cls = matched_type()
    scheduler = diffsci.models.EDMScheduler()
    rhs = lambda x, t: t*x/(.25+t*t)
    method = cls(0.)
    base = diffsci.models.KarrasIntegrator(s_schurn=0, s_tmin=None, churn_cap=None)
    a = b = torch.ones(30, 1)
    times = torch.linspace(2., .1, 20)
    for t, dt in zip(times[:-1], times.diff()):
        a = method.step(a, t, dt, rhs, scheduler.scheduler_fns)
        b = base.step(b, t, dt, rhs, scheduler.scheduler_fns)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    method = cls(2.)
    outputs = []
    for _ in range(2):
        method.reset()
        torch.manual_seed(5)
        outputs.append(method.step(torch.ones(30, 1), torch.tensor(1.), torch.tensor(-.1), rhs,
                                   scheduler.scheduler_fns))
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    with pytest.raises(ValueError, match='reset'):
        method.step(torch.ones(3, 1), torch.tensor(1.), torch.tensor(-.1), rhs, scheduler.scheduler_fns)
    method.reset()
    assert method._previous_noise is None
