"""Plot matched residual interventions and their Gaussian prediction gaps."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'diffsci-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .residual_ablation import binned_kl


def read_rows(path):
    with Path(path).open() as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key,value in row.items():
            if key in ('run','kl_direction'):
                continue
            row[key]=(value=='True') if value in ('True','False') else float(value)
    return rows


def corr(x,y):
    return float(spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else None


def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=160,bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def create_plots(output_dir,primary_bins=64):
    out=Path(output_dir)
    meta=json.loads((out/'metadata.json').read_text())
    if not meta['complete']:
        raise ValueError('Complete the ablation before making aggregate plots')
    effects=read_rows(out/'paired_effects.csv')
    results=read_rows(out/'ablation_results.csv')
    if primary_bins not in meta['bins']:
        primary_bins=meta['bins'][len(meta['bins'])//2]
    positive=[g for g in meta['gammas'] if g>0]
    gammas=sorted(set(min(positive,key=lambda g:abs(g-target)) for target in (.2,1.,5.)))
    source=out.parent/'score_error_modes/checkpoint_summary.csv'
    if not source.exists():
        source=Path(meta['root'])/'paper_odds/outputs/score_error_modes/checkpoint_summary.csv'
    with source.open() as f:
        residual={(r['run'],int(r['epoch'])):float(r['nonlinear_energy_fraction']) for r in csv.DictReader(f)}
    summary=dict(checkpoints=meta['completed_checkpoints'],primary_bins=primary_bins,
                 pooled_particles=meta['particles']*len(meta['seeds']),groups={},
                 convention='positive disagreement reduction means removing the residual improves agreement',
                 uncertainty='Monte Carlo only; checkpoints from three runs are not independent replicates')
    for bins in meta['bins']:
        for gamma in gammas:
            for direction in ('q_p','p_q'):
                group=[r for r in effects if r['bins']==bins and r['gamma']==gamma and r['kl_direction']==direction]
                gain=np.array([r['disagreement_reduction'] for r in group])
                effect=np.array([r['nonlinear_effect_log10_ratio'] for r in group])
                resolved=np.array([r['both_arms_above_control_floor'] for r in group])
                prediction=[r['gaussian_profile_log10_ratio'] for r in group]
                moment_prediction=[r['gaussian_moment_log10_ratio'] for r in group]
                full=[r['full_log10_ratio'] for r in group]
                affine=[r['affine_log10_ratio'] for r in group]
                residual_fraction=[residual[(r['run'],int(r['epoch']))] for r in group]
                stat=dict(count=len(group),above_control_floor_count=int(resolved.sum()),
                          fraction_disagreement_reduced=float(np.mean(gain>0)),median_disagreement_reduction=float(np.median(gain)),
                          fraction_disagreement_reduced_above_floor=float(np.mean(gain[resolved]>0)) if resolved.any() else None,
                          median_abs_nonlinear_effect=float(np.median(abs(effect))),
                          median_effect_mc_se=float(np.median([r['nonlinear_effect_mc_se'] for r in group])),
                          median_full_profile_disagreement=float(np.median([r['full_profile_disagreement'] for r in group])),
                          median_affine_profile_disagreement=float(np.median([r['affine_profile_disagreement'] for r in group])),
                          median_affine_moment_disagreement=float(np.median([r['affine_gaussian_moment_disagreement'] for r in group])),
                          spearman_full_profile=corr(full,prediction),spearman_affine_profile=corr(affine,prediction),
                          spearman_affine_gaussian_moments=corr(affine,moment_prediction),
                          spearman_residual_fraction_absolute_effect=corr(residual_fraction,abs(effect)),by_run={})
                for run in sorted(set(r['run'] for r in group)):
                    subset=[r for r in group if r['run']==run]
                    stat['by_run'][run]=dict(count=len(subset),median_disagreement_reduction=float(np.median([r['disagreement_reduction'] for r in subset])),
                                             fraction_disagreement_reduced=float(np.mean([r['disagreement_reduction']>0 for r in subset])))
                summary['groups'][f'bins{bins}_gamma{gamma:g}_{direction}']=stat
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,len(gammas),figsize=(5*len(gammas),8),squeeze=False,layout='constrained')
    effect_fig,effect_axes=plt.subplots(2,len(gammas),figsize=(5*len(gammas),8),squeeze=False,layout='constrained')
    for j,gamma in enumerate(gammas):
        for i,direction in enumerate(('q_p','p_q')):
            group=[r for r in effects if r['bins']==primary_bins and r['gamma']==gamma and r['kl_direction']==direction]
            ax=axes[i,j];effect_ax=effect_axes[i,j]
            for run in sorted(set(r['run'] for r in group)):
                subset=[r for r in group if r['run']==run]
                x=np.array([r['full_profile_disagreement'] for r in subset]);y=np.array([r['affine_profile_disagreement'] for r in subset])
                unresolved=np.array([not r['both_arms_above_control_floor'] for r in subset])
                artist=ax.scatter(x,y,s=25,label=run,alpha=.8)
                color=artist.get_facecolor()[0]
                ax.scatter(x[unresolved],y[unresolved],s=8,c='white',zorder=4)
                xx=[residual[(r['run'],int(r['epoch']))] for r in subset]
                yy=[r['nonlinear_effect_log10_ratio'] for r in subset]
                error=[r['nonlinear_effect_mc_se'] for r in subset]
                effect_ax.errorbar(xx,yy,yerr=error,fmt='o',ms=3.5,lw=.6,alpha=.65,color=color,label=run)
            lim=max([r['full_profile_disagreement'] for r in group]+[r['affine_profile_disagreement'] for r in group])*1.05
            ax.plot([0,lim],[0,lim],'--',c='.5',lw=1)
            ax.set(xlabel='Full score: absolute prediction gap',ylabel='Affine error only: absolute prediction gap',
                   title=f'gamma={gamma:g}; KL({direction.replace("_", " || ")})',xlim=(-.01,lim),ylim=(-.01,lim))
            ax.legend(fontsize=8)
            effect_ax.axhline(0,c='.5',lw=1)
            effect_ax.set(xlabel='Nonlinear fraction of integrated score-error energy',
                          ylabel='Full minus affine log10(KL_gamma / KL_ODE)',title=f'gamma={gamma:g}; {direction}')
            effect_ax.legend(fontsize=8)
    fig.suptitle(f'Does removing the nonlinear residual improve the Gaussian profile prediction? ({primary_bins} bins)\n'
                 'Below diagonal: improved agreement. White centers: at least one KL is within 3 times the exact-score control.',fontsize=12)
    effect_fig.suptitle('Controlled effect of retaining the nonlinear residual\n'
                        'Negative: the residual makes stochasticity more favorable relative to the ODE. Bars: paired Monte Carlo standard errors.',fontsize=12)
    save(fig,out,'ablation_prediction_gaps');save(effect_fig,out,'nonlinear_effect_vs_energy')
    # Same checkpoint epoch in each run, or its closest available checkpoint.
    selected=[]
    for run in sorted(set(r['run'] for r in results)):
        epochs=sorted(set(int(r['epoch']) for r in results if r['run']==run))
        selected.append((run,min(epochs,key=lambda e:abs(e-24))))
    fig,axes=plt.subplots(2,len(selected),figsize=(5*len(selected),8),squeeze=False,layout='constrained')
    for j,(run,epoch) in enumerate(selected):
        for i,direction in enumerate(('q_p','p_q')):
            ax=axes[i,j]
            for lam,label in [(0.,'Affine error only'),(1.,'Full learned score')]:
                subset=sorted([r for r in results if r['run']==run and r['epoch']==epoch and r['bins']==primary_bins and r['residual_lambda']==lam],key=lambda r:r['gamma'])
                ax.plot([r['gamma'] for r in subset],[r[f'log10_ratio_ode_{direction}'] for r in subset],'o-',ms=4,label=label)
            for column,label,style in [('gaussian_profile_log10_ratio_ode_'+direction,'Gaussian first-order response','--'),
                                       ('gaussian_moment_log10_ratio_ode_'+direction,'Gaussian full moments',':')]:
                ax.plot([r['gamma'] for r in subset],[r[column] for r in subset],style,label=label)
            ax.axhline(0,c='.5',lw=.8);ax.set_xscale('symlog',linthresh=.01)
            ax.set(xlabel='Stochasticity gamma',ylabel='log10(KL_gamma / KL_ODE)',title=f'{run}, epoch {epoch}; {direction}')
            ax.legend(fontsize=8)
    save(fig,out,'ablation_example_curves')
    with np.load(out/'exact_control.npz') as data:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for bins in meta['bins']:
            control=binned_kl(data[f'counts_{bins}'][:,0].sum(axis=1),meta['pseudocount'])
            for i,ax in enumerate(axes):
                ax.plot(meta['gammas'],control[:,i],'o-',label=f'{bins} bins')
                ax.set_xscale('symlog',linthresh=.01)
                ax.set(xlabel='gamma',ylabel='Exact-score control: estimated KL',title=('KL(q || p)' if i==0 else 'KL(p || q)'))
                ax.legend()
        fig.suptitle('Exact-score control: continuous-time exact KL is zero for the exact prior')
        save(fig,out,'exact_score_control')
    summary['max_score_table_relative_rms_error']=max(r['table_relative_rms_error_max'] for r in results)
    summary['max_direct_fallback_points_per_checkpoint']=int(max(r['fallback_points'] for r in results))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Saved ablation figures and summary to {out}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output_dir',type=Path)
    p.add_argument('--primary-bins',type=int,default=64)
    args=p.parse_args()
    create_plots(args.output_dir,args.primary_bins)
