"""Check time-grid and quadrature sensitivity at three epochs of every run.

Run after fit_checkpoint_modes:
    python -m paper_odds.score_error_analysis.validate_checkpoint_modes
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint_score import CheckpointScore
from .fit_checkpoint_modes import DEFAULT_OUTPUT, ROOT, project_grid
from .score_error_modes import (
    GaussianMixture1D, fit_exponential, integration_weights,
    normalized_profiles, profile_response, variance_clock,
)


def validate_resolution(output_dir, root=ROOT):
    out = Path(output_dir)
    meta = json.loads((out/'metadata.json').read_text())
    if not meta['complete']:
        raise ValueError('Complete the checkpoint analysis before validating resolution')
    with (out/'checkpoint_summary.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    mixture = GaussianMixture1D(**meta['mixture'])
    clock = variance_clock(mixture, meta['sigma_min'], meta['sigma_max'],
                           2*meta['time_points']-1, 2*meta['sigma_points']-1)
    selected = []
    for run in meta['runs']:
        group = sorted([r for r in rows if r['run'] == run], key=lambda r: int(r['epoch']))
        selected.extend(group[i] for i in sorted({0, len(group)//2, len(group)-1}))
    checks = []
    for row in selected:
        model = CheckpointScore(Path(root)/row['checkpoint'])
        arrays = project_grid(mixture, clock['sigma'], model, 2*meta['quadrature_order'], 8192)
        arrays.update(normalized_profiles(arrays['b'], arrays['C'], clock['variance'], clock['reference_variance']))
        iw = integration_weights(clock['distance'])
        residual_fraction = float(iw@arrays['residual_energy']/(iw@arrays['total_energy']))
        with np.load(out/'profiles'/f"{row['run']}_epoch{int(row['epoch']):02d}.npz") as base:
            response = profile_response(clock['distance'], arrays['a_linear'], arrays['u_linear'], base['prediction_gamma'])
            gamma = base['prediction_gamma']
            baseline = int(np.flatnonzero(gamma == base['empirical_gamma_values'][0])[0])
            difference = np.log10(response['kl']/response['kl'][baseline])-np.log10(base['profile_kl']/base['profile_kl'][baseline])
            combined = np.array([response['shape_response'], response['mean_response']]).T
            previous = np.array([base['profile_shape_response'], base['profile_mean_response']]).T
            response_relative = float(np.linalg.norm(combined-previous)/max(np.linalg.norm(combined), 1e-30))
        check = dict(run=row['run'], epoch=int(row['epoch']),
                     nonlinear_fraction_absolute_change=abs(residual_fraction-float(row['nonlinear_energy_fraction'])),
                     combined_response_relative_change=response_relative,
                     max_abs_log10_kl_ratio_change=float(np.max(np.abs(difference))))
        fit_lo, fit_hi = meta['fit_distance']
        fit_hi = np.inf if fit_hi == 'infinity' else fit_hi
        mask = (clock['distance'] >= fit_lo) & (clock['distance'] <= fit_hi)
        for mode, key in (('a', 'a_linear'), ('m', 'u_linear')):
            fit = fit_exponential(clock['distance'][mask], arrays[key][mask], meta['kappa_bounds'])
            check[f'refined_kappa_{mode}'] = fit['kappa']
            check[f'kappa_{mode}_scaled_change'] = abs(fit['kappa']-float(row[f'kappa_{mode}']))/(1+abs(fit['kappa']))
        checks.append(check)
        print(f"{row['run']} epoch {row['epoch']}: response change {response_relative:.3%}, "
              f"max log10 ratio change {check['max_abs_log10_kl_ratio_change']:.4g}", flush=True)
    maxima = {key: max(c[key] for c in checks) for key in (
        'nonlinear_fraction_absolute_change', 'combined_response_relative_change',
        'max_abs_log10_kl_ratio_change', 'kappa_a_scaled_change', 'kappa_m_scaled_change')}
    passed = (maxima['nonlinear_fraction_absolute_change'] < .005
              and maxima['combined_response_relative_change'] < .02
              and maxima['max_abs_log10_kl_ratio_change'] < .05
              and maxima['kappa_a_scaled_change'] < .05
              and maxima['kappa_m_scaled_change'] < .05)
    result = dict(passed=passed, checkpoints=len(checks), primary_fingerprint=meta['fingerprint'],
                  refined_quadrature_order=2*meta['quadrature_order'],
                  refined_time_points=2*meta['time_points']-1, refined_sigma_points=2*meta['sigma_points']-1,
                  thresholds=dict(nonlinear_fraction_absolute_change=.005, combined_response_relative_change=.02,
                                  max_abs_log10_kl_ratio_change=.05, kappa_scaled_change=.05),
                  maxima=maxima, checks=checks)
    (out/'resolution_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    if not passed:
        raise AssertionError(f'Resolution sensitivity exceeds tolerances: {maxima}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = validate_resolution(args.output_dir, args.root)
    print(f"Resolution validation passed on {result['checkpoints']} representative checkpoints.")
