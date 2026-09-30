import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = (Path(__file__).resolve().parents[1] /
        'notebooks/exploratory/bps/karras_integrator/histogram_metrics.py')
spec = importlib.util.spec_from_file_location('histogram_metrics', path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def test_other_sampler_outliers_cannot_change_a_methods_kl():
    reference = np.linspace(-1, 1, 1000)
    candidate = reference + 0.2
    alone = metrics.reference_histogram_kl({'candidate': candidate}, reference)
    together = metrics.reference_histogram_kl(
        {'candidate': candidate, 'unstable': np.full(1000, -1e12)}, reference)
    np.testing.assert_array_equal(alone['candidate'], together['candidate'])
    assert np.all(together['candidate'] > 0)
    assert np.all(together['unstable'] > 10)


def test_both_tail_bins_count_every_sample_without_clipping():
    reference = np.array([-1., 0., 1.])
    edges = metrics.reference_histogram_edges(reference, nbins=2)
    assert len(edges) == 5  # Two interior bins and two tail bins.
    counts = np.histogram([-1e12, -1, 0, 1, 1e12], bins=edges)[0]
    np.testing.assert_array_equal(counts, [1, 1, 2, 1])
    reference_counts = np.histogram(reference, bins=edges)[0]
    assert reference_counts[0] == reference_counts[-1] == 0


def test_identical_and_constant_samples_have_zero_kl():
    for reference in [np.ones(20), np.linspace(-1, 1, 100)]:
        result = metrics.reference_histogram_kl({'same': reference}, reference)
        np.testing.assert_array_equal(result['same'], [0., 0.])


def test_explicit_partition_matches_automatic_reference_partition():
    reference = np.linspace(-1, 1, 1000)
    samples = {'shifted': reference + .1}
    edges = metrics.reference_histogram_edges(reference)
    auto = metrics.reference_histogram_kl(samples, reference)
    fixed = metrics.reference_histogram_kl(samples, reference, edges=edges)
    np.testing.assert_array_equal(auto['shifted'], fixed['shifted'])


@pytest.mark.parametrize('values', [[], [np.nan], [np.inf]])
def test_invalid_samples_are_not_silently_discarded(values):
    with pytest.raises(ValueError, match='nonempty and finite'):
        metrics.reference_histogram_kl({'bad': values}, [0., 1.])
