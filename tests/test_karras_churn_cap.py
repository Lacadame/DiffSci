import numpy as np
import pytest
import torch

from diffsci.models import EDMScheduler, KarrasIntegrator


@pytest.mark.parametrize("cap, expected_churn", [
    (np.sqrt(2)-1, np.sqrt(2)-1), (None, 2.0), (0.0, 0.0),
])
def test_churn_controls_noise_injection_and_evaluation_time(cap, expected_churn):
    x = torch.zeros(32, 1)
    scheduler = EDMScheduler()
    integrator = KarrasIntegrator(s_schurn=20, s_tmin=None, s_noise=1,
                                  churn_cap=cap)
    times = []

    def rhs(x, t):
        times.append(float(t))
        return torch.zeros_like(x)

    torch.manual_seed(42)
    noise = torch.randn_like(x)
    torch.manual_seed(42)
    result = integrator.step(x, torch.tensor(1.), torch.tensor(-0.1), rhs,
                             scheduler.scheduler_fns, nsteps=10)
    torch.testing.assert_close(result, noise*np.sqrt((1+expected_churn)**2-1))
    assert times == pytest.approx([1+expected_churn, 0.9])


def test_uncapped_matches_default_below_cap_and_respects_window():
    x = torch.ones(32, 1)
    scheduler = EDMScheduler()
    for churn, window in [(1.0, (None, None)), (20.0, (2.0, 3.0))]:
        results = []
        for integrator in [KarrasIntegrator(s_schurn=churn, s_tmin=window[0],
                                            s_tmax=window[1]),
                           KarrasIntegrator(s_schurn=churn, s_tmin=window[0],
                                            s_tmax=window[1], churn_cap=None)]:
            torch.manual_seed(42)
            results.append(integrator.step(x, torch.tensor(1.), torch.tensor(-0.1),
                                           lambda x, t: -x, scheduler.scheduler_fns,
                                           nsteps=10))
        torch.testing.assert_close(*results, rtol=0, atol=0)


@pytest.mark.parametrize("cap", [-1, float('nan'), float('inf')])
def test_invalid_cap(cap):
    with pytest.raises(ValueError, match="churn_cap"):
        KarrasIntegrator(churn_cap=cap)
