"""Evaluate numerical agreement of all saved kappa strategies on matched pairs.

Predictions are scored against the identity line, without fitting a calibration
slope or intercept. Correlations remain separate descriptive diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .compare_exponential_predictions import PAPER, read_csv, write_csv
from .plot_checkpoint_modes import plt, save_figure


LABELS = {
    'profile': 'Signed profiles',
    'exponential_finite': 'Original exponential (finite)',
    'exponential_infinite': 'Original exponential (infinite)',
    'smooth_0.01': 'Smoothed 0.01', 'smooth_0.05': 'Smoothed 0.05', 'smooth_0.2': 'Smoothed 0.2',
    'cumulative': 'Cumulative response',
    'loglog': 'Log-log, clock weighted', 'loglog_unweighted': 'Log-log, ordinary',
    'loglog_cutoff_1e-8': 'Log-log, cutoff 1e-8', 'loglog_cutoff_1e-4': 'Log-log, cutoff 1e-4',
    'no_change': 'Baseline: KL ratio = 1',
}
KEYS = ('dataset', 'run', 'epoch', 'gamma', 'baseline_gamma', 'bins', 'kl_direction')


def safe_divide(numerator, denominator):
    return float(numerator/denominator) if denominator > 0 else np.nan


def score_predictions(predicted_log10, measured_log10):
    """Fixed-prediction losses; log errors treat reciprocal ratio errors equally."""
    x, y = np.asarray(predicted_log10, float), np.asarray(measured_log10, float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Require matched finite one-dimensional prediction/measurement arrays')
    if len(x) == 0:
        # Preserve the schema for an empty control-floor/run subset.
        result = score_predictions([0., 0.], [0., 0.])
        return {k: 0 if k == 'n' else np.nan for k in result}
    with np.errstate(over='raise', invalid='raise'):
        ratio_x, ratio_y = 10.**x, 10.**y
    if np.any(ratio_x <= 0) or np.any(ratio_y <= 0):
        raise ValueError('Ratios must be representable and positive')
    result = dict(n=len(x))
    for scale, p, m, baseline in [('log10', x, y, 0.), ('ratio', ratio_x, ratio_y, 1.)]:
        error = p-m
        mse, baseline_mse = float(np.mean(error**2)), float(np.mean((m-baseline)**2))
        pmean, mmean = float(p.mean()), float(m.mean())
        pvar, mvar = float(np.mean((p-pmean)**2)), float(np.mean((m-mmean)**2))
        covariance = float(np.mean((p-pmean)*(m-mmean)))
        denominator = pvar+mvar+(pmean-mmean)**2
        result.update({
            f'mse_{scale}': mse, f'rmse_{scale}': float(np.sqrt(mse)),
            f'mae_{scale}': float(np.mean(abs(error))), f'bias_{scale}': float(error.mean()),
            f'median_absolute_error_{scale}': float(np.median(abs(error))),
            f'p90_absolute_error_{scale}': float(np.quantile(abs(error), .9)),
            f'baseline_mse_{scale}': baseline_mse,
            f'skill_vs_no_change_{scale}': 1-safe_divide(mse, baseline_mse),
            f'measured_variance_{scale}': mvar,
            f'identity_r2_{scale}': 1-safe_divide(mse, mvar),
            # Identical constants are perfect agreement; Pearson is undefined.
            f'concordance_{scale}': float(2*covariance/denominator) if denominator > 0 else 1.,
            f'pearson_{scale}': float(pearsonr(p, m).statistic) if len(p) > 2 and pvar > 0 and mvar > 0 else np.nan,
        })
    result['spearman'] = float(spearmanr(x, y).statistic) if len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
    result['sign_agreement'] = float(np.mean(np.sign(x) == np.sign(y)))
    return result


def load_predictions(paper):
    """Join by checkpoint and exact measurement protocol; reject inconsistent repeats."""
    records = {}
    paths = [paper/p for p in ('outputs/exponential_kl_comparison/predictions_vs_measurements.csv',
                              'outputs/kappa_sensitivity/predictions_vs_measurements.csv',
                              'outputs/loglog_kappa/predictions_vs_measurements.csv')]
    for path in paths:
        for row in read_csv(path):
            if int(row['bins']) not in (0, 64):
                continue
            base = {k: (int(row[k]) if k in ('epoch', 'bins') else float(row[k])
                        if k in ('gamma', 'baseline_gamma') else row[k]) for k in KEYS}
            key = tuple(base[k] for k in KEYS)
            measured = float(row['measured_log10_ratio'])
            floor = '' if row['above_control_floor'] == '' else row['above_control_floor'] == 'True'
            if key not in records:
                records[key] = dict(**base, measured_log10_ratio=measured, above_control_floor=floor, predictions={})
            record = records[key]
            if not np.isclose(record['measured_log10_ratio'], measured, rtol=0, atol=1e-12) or record['above_control_floor'] != floor:
                raise ValueError(f'Inconsistent measurement or floor flag: {key}')
            predictions = {'profile': float(row['profile_log10_ratio'])}
            if 'method' in row:
                method = 'exponential_finite' if row['method'] == 'original' else row['method']
                predictions[method] = float(row['predicted_log10_ratio'])
            else:
                predictions.update({method: float(row[f'{method}_log10_ratio'])
                                    for method in ('exponential_finite', 'exponential_infinite')})
            for method, prediction in predictions.items():
                if method in record['predictions'] and not np.isclose(record['predictions'][method], prediction, rtol=0, atol=1e-12, equal_nan=True):
                    raise ValueError(f'Inconsistent repeated prediction: {key}, {method}')
                record['predictions'][method] = prediction
    for record in records.values():
        record['predictions']['no_change'] = 0.
        if set(record['predictions']) != set(LABELS):
            raise ValueError(f'Missing or extra methods in {record}')
        values = np.array([record['measured_log10_ratio'], *record['predictions'].values()])
        record['common_finite'] = bool(np.isfinite(values).all())
        if record['common_finite']:
            with np.errstate(over='ignore', under='ignore'):
                ratios = 10.**values
            if not np.isfinite(ratios).all() or np.any(ratios <= 0):
                raise ValueError('Raw ratios overflow/underflow; do not silently exclude them')
    if not records:
        raise ValueError('No comparisons found')
    return list(records.values()), paths


def evaluate(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['dataset'], r['gamma'], r['baseline_gamma'], r['bins'], r['kl_direction']].append(r)
    results = []
    for (dataset, gamma, baseline, bins, direction), group in sorted(groups.items()):
        for subset in (('all',) if dataset == 'saved_sweep' else ('all', 'above_control_floor')):
            for run in ('all', *sorted(set(r['run'] for r in group))):
                selected = [r for r in group if (run == 'all' or r['run'] == run)
                            and (subset == 'all' or r['above_control_floor'])]
                common = [r for r in selected if r['common_finite']]
                y = [r['measured_log10_ratio'] for r in common]
                for method in LABELS:
                    metrics = score_predictions([r['predictions'][method] for r in common], y)
                    results.append(dict(dataset=dataset, gamma=gamma, baseline_gamma=baseline, bins=bins,
                        kl_direction=direction, subset=subset, run=run, method=method,
                        available_count=len(selected), excluded_common_nonfinite=len(selected)-len(common), **metrics))
    return results


def aggregate(results):
    """Equal weight per gamma; correlations are averaged within gamma, never pooled."""
    groups = defaultdict(list)
    for r in results:
        groups[r['dataset'], r['bins'], r['kl_direction'], r['subset'], r['run'], r['method']].append(r)
    rows = []
    for (dataset, bins, direction, subset, run, method), group in sorted(groups.items()):
        row = dict(dataset=dataset, bins=bins, kl_direction=direction, subset=subset, run=run, method=method,
                   gamma_count=len(group), gammas=json.dumps(sorted(r['gamma'] for r in group)),
                   n_pairs=sum(r['n'] for r in group))
        for scale in ('log10', 'ratio'):
            for metric in ('mse', 'mae', 'bias', 'baseline_mse', 'measured_variance'):
                row[f'{metric}_{scale}'] = float(np.mean([r[f'{metric}_{scale}'] for r in group]))
            row[f'rmse_{scale}'] = float(np.sqrt(row[f'mse_{scale}']))
            row[f'skill_vs_no_change_{scale}'] = 1-safe_divide(row[f'mse_{scale}'], row[f'baseline_mse_{scale}'])
            row[f'within_gamma_identity_r2_{scale}'] = 1-safe_divide(row[f'mse_{scale}'], row[f'measured_variance_{scale}'])
            for metric in ('pearson', 'concordance'):
                row[f'mean_{metric}_{scale}'] = float(np.mean([r[f'{metric}_{scale}'] for r in group]))
        row['mean_spearman'] = float(np.mean([r['spearman'] for r in group]))
        rows.append(row)
    return rows


def identity_figure(records, dataset='controlled_full', gamma=1., direction='q_p', subset='all'):
    methods = ['profile', 'exponential_finite', 'smooth_0.2', 'cumulative', 'loglog', 'loglog_unweighted']
    selected = [r for r in records if r['dataset'] == dataset and r['gamma'] == gamma
                and r['kl_direction'] == direction and r['common_finite']
                and (subset == 'all' or r['above_control_floor'])]
    if not selected:
        raise ValueError('No records for the requested exact gamma and dataset')
    y = np.array([r['measured_log10_ratio'] for r in selected])
    predictions = np.array([[r['predictions'][method] for r in selected] for method in methods])
    low, high = min(y.min(), predictions.min()), max(y.max(), predictions.max())
    pad = max(.03, .05*(high-low)); bounds = (low-pad, high+pad)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout='constrained', sharex=True, sharey=True)
    for ax, method, prediction in zip(axes.flat, methods, predictions):
        for i, run in enumerate(sorted(set(r['run'] for r in selected))):
            mask = np.array([r['run'] == run for r in selected])
            ax.scatter(prediction[mask], y[mask], s=19, alpha=.7, c=f'C{i}', label=run)
        metrics = score_predictions(prediction, y)
        ax.plot(bounds, bounds, '--', c='.3', lw=1)
        ax.axhline(0, c='.8', lw=.6); ax.axvline(0, c='.8', lw=.6)
        ax.set(xlim=bounds, ylim=bounds, aspect='equal', xlabel='Predicted log10 KL ratio', ylabel='Measured log10 KL ratio',
               title=f"{LABELS[method]}\nRMSE={metrics['rmse_log10']:.3f}; bias={metrics['bias_log10']:+.3f}; rho={metrics['spearman']:+.3f}")
    axes[0, 0].legend(fontsize=8)
    zero = score_predictions(np.zeros_like(y), y)['rmse_log10']
    fig.suptitle(f'{dataset}: gamma={gamma:g}, {direction}, {subset}, n={len(y)}; ratio=1 baseline RMSE={zero:.3f}\n'
                 'Agreement is distance from the identity line; no slope or intercept fitted to measurements. Shared axes.', fontsize=12)
    return fig


def figures(results, records, out):
    from matplotlib.colors import LogNorm
    selected = [r for r in results if r['dataset'] == 'controlled_full' and r['run'] == 'all' and r['subset'] == 'all']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout='constrained')
    raw_fig, raw_axes = plt.subplots(1, 2, figsize=(16, 6), layout='constrained')
    gammas, methods = [.2, 1., 5.], list(LABELS)
    max_rmse = max(r['rmse_log10'] for r in selected)
    raw_values = [r['rmse_ratio'] for r in selected]
    for j, direction in enumerate(('q_p', 'p_q')):
        group = {(r['method'], r['gamma']): r for r in selected if r['kl_direction'] == direction}
        for i, metric in enumerate(('rmse_log10', 'spearman')):
            ax = axes[i, j]
            values = np.array([[group[m, g][metric] for g in gammas] for m in methods])
            cmap = plt.get_cmap('Blues' if i == 0 else 'RdBu_r').with_extremes(bad='#dddddd')
            mesh = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0 if i == 0 else -1, vmax=max_rmse if i == 0 else 1, aspect='auto')
            for index, value in np.ndenumerate(values):
                ax.text(index[1], index[0], f'{value:.3f}' if np.isfinite(value) else '—', ha='center', va='center',
                        color='white' if np.isfinite(value) and (value > .55*max_rmse if i == 0 else abs(value) > .6) else 'black', fontsize=9)
            ax.set_xticks(range(3), [f'gamma={g:g}' for g in gammas])
            ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=9)
            ax.set_title(f'{direction}: '+('RMSE of log10 ratios (lower is better)' if i == 0 else 'Spearman rank correlation'))
            fig.colorbar(mesh, ax=ax, fraction=.025)
        values = np.array([[group[m, g]['rmse_ratio'] for g in gammas] for m in methods])
        ax = raw_axes[j]
        mesh = ax.imshow(values, cmap='Blues', norm=LogNorm(min(raw_values), max(raw_values)), aspect='auto')
        for index, value in np.ndenumerate(values):
            ax.text(index[1], index[0], f'{value:.3g}', ha='center', va='center', fontsize=9,
                    color='white' if value > np.sqrt(min(raw_values)*max(raw_values)) else 'black')
        ax.set_xticks(range(3), [f'gamma={g:g}' for g in gammas])
        ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=9)
        ax.set_title(f'{direction}: RMSE of raw KL ratios (lower is better)')
        raw_fig.colorbar(mesh, ax=ax, fraction=.025, label='RMSE; logarithmic color scale')
    fig.suptitle('Prediction agreement and correlation measure different properties\nFull learned score; 144 matched checkpoints; 64 bins; measured ODE baseline', fontsize=13)
    raw_fig.suptitle('Agreement on raw ratios: large ratio errors receive more weight\nAll values retained; raw RMSE annotated, logarithmic colors for readability', fontsize=13)
    save_figure(fig, out, 'agreement_and_correlation')
    save_figure(raw_fig, out, 'raw_ratio_agreement')
    for direction in ('q_p', 'p_q'):
        save_figure(identity_figure(records, direction=direction), out, f'identity_gamma1_{direction}')


def run(paper=PAPER, output_dir=None):
    paper = Path(paper); out = Path(output_dir) if output_dir else paper/'outputs/prediction_agreement'
    out.mkdir(parents=True, exist_ok=True)
    records, paths = load_predictions(paper)
    results = evaluate(records); aggregates = aggregate(results)
    write_csv(out/'metrics.csv', results)
    write_csv(out/'aggregate_metrics.csv', aggregates)
    errors = []
    for r in records:
        for method, prediction in r['predictions'].items():
            e = prediction-r['measured_log10_ratio']
            errors.append({k: v for k, v in r.items() if k != 'predictions'} | dict(method=method,
                predicted_log10_ratio=prediction, error_log10=e, squared_error_log10=e*e,
                predicted_ratio=10.**prediction, measured_ratio=10.**r['measured_log10_ratio'],
                squared_error_ratio=(10.**prediction-10.**r['measured_log10_ratio'])**2))
    write_csv(out/'checkpoint_errors.csv', errors)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    figures(results, records, out)
    paths += [Path(__file__), paper/'score_error_analysis/compare_exponential_predictions.py',
              paper/'score_error_analysis/plot_checkpoint_modes.py']
    metadata = dict(complete=True, measurement_pairs=len(records), methods=list(LABELS),
        per_gamma_metric_rows=len(results), aggregate_metric_rows=len(aggregates),
        nonfinite_common_exclusions=sum(not r['common_finite'] for r in records),
        agreement='Fixed predictions against identity line; no fitted calibration; no kappa refitting',
        main_metric='RMSE of log10 KL ratios; raw-ratio MSE/RMSE/MAE also reported',
        baseline='Predict KL_gamma/KL_baseline=1 for every checkpoint; log prediction=0',
        skill='1 - MSE_model/MSE_no_change; positive means improvement over ratio=1',
        r2='1 - MSE/variance(measured); fixed-prediction R², not squared Pearson correlation',
        concordance='2*cov(predicted,measured)/(var(predicted)+var(measured)+(mean(predicted)-mean(measured))²)',
        aggregate='Equal gamma weights for losses; average within-gamma correlations, never pooled across gammas',
        scope='Original sweeps keep gamma=0.01 baseline; controlled full/affine use measured ODE baseline and 64 bins',
        inference='Descriptive evaluation on three correlated training trajectories; no independent-checkpoint p-values or hyperparameter tuning',
        source_sha256={str(p.relative_to(paper)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved {len(results)} per-gamma and {len(aggregates)} aggregate metric rows for {len(records)} matched pairs.', flush=True)
    for direction in ('q_p', 'p_q'):
        group = [r for r in aggregates if r['dataset'] == 'controlled_full' and r['run'] == 'all'
                 and r['subset'] == 'all' and r['kl_direction'] == direction]
        for r in sorted(group, key=lambda r: r['rmse_log10']):
            print(f"{direction} {r['method']}: RMSE(log10)={r['rmse_log10']:.4f}, "
                  f"skill={r['skill_vs_no_change_log10']:+.3f}, mean rho={r['mean_spearman']:+.3f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path, default=PAPER)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run(args.paper_dir, args.output_dir)
