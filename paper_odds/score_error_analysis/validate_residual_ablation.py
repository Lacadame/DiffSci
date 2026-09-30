"""Paired numerical refinement with Brownian bridges on nested EDM grids.

Doubles integration steps, score-table nodes, and quadrature on selected models.
Each pair of fine Brownian increments sums to the original coarse increment,
so resolution differences are not obscured by a new independent path sample.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from .checkpoint_score import CheckpointScore
from .score_error_modes import GaussianMixture1D
from .residual_ablation import binned_kl, brownian_generators, histogram_counts, sampling_grid, shared_initial_states, target_partition
from .run_residual_ablation import DEFAULT_ABLATION_OUTPUT, build_field


def split_brownian_increment(coarse, bridge_normal, h1, h2):
    first=h1/(h1+h2)*coarse+np.sqrt(h1*h2/(h1+h2))*bridge_normal
    return first,coarse-first


def sample_coupled_refinement(field,gammas,lambdas,initial,seeds,coarse_clock,integrator='heun'):
    if integrator not in ('euler', 'heun'):
        raise ValueError('Integrator must be euler or heun')
    fine_ell=field.clock['ell']
    np.testing.assert_allclose(fine_ell[::2],coarse_clock['ell'],rtol=1e-12,atol=1e-12)
    z=np.broadcast_to(initial,(len(gammas),len(lambdas),*initial.shape)).copy()
    gamma=np.asarray(gammas)[:,None,None];sqrt_gamma=np.sqrt(gamma)
    base_rngs=brownian_generators(seeds)
    bridge_rngs=[np.random.default_rng(np.random.SeedSequence([int(seed),4909])) for seed in seeds]
    for i,H in enumerate(np.diff(coarse_clock['ell'])):
        coarse=np.stack([rng.standard_normal(initial.shape[-1]) for rng in base_rngs])*np.sqrt(H)
        bridge=np.stack([rng.standard_normal(initial.shape[-1]) for rng in bridge_rngs])
        h1,h2=np.diff(fine_ell[2*i:2*i+3])
        increments=split_brownian_increment(coarse,bridge,h1,h2)
        for half,(h,increment) in enumerate(zip((h1,h2),increments)):
            index=2*i+half
            noise=sqrt_gamma*increment
            for j,lam in enumerate(lambdas):
                state=z[:,j]
                drift0=state/2+(1+gamma)/2*field.score(state,index,lam)
                predictor=state+h*drift0+noise
                if integrator == 'euler':
                    z[:,j]=predictor
                else:
                    drift1=predictor/2+(1+gamma)/2*field.score(predictor,index+1,lam)
                    z[:,j]=state+h/2*(drift0+drift1)+noise
        if not np.isfinite(z).all():
            raise FloatingPointError('Nonfinite refined trajectory')
    return field.mixture.mean+field.clock['sqrt_variance'][-1]*z


def validate_one(record,meta,out):
    torch.set_num_threads(1)
    out=Path(out)
    mixture=GaussianMixture1D(**meta['mixture'])
    refined=dict(meta,steps=meta['steps']*2,table_nodes=meta['table_nodes']*2-1,quadrature_order=meta['quadrature_order']*2)
    model=CheckpointScore(Path(meta['root'])/record['checkpoint'])
    field,_,table_validation=build_field(model,mixture,refined)
    coarse_clock=sampling_grid(mixture,meta['sigma_min'],meta['sigma_max'],meta['steps'])
    initial=shared_initial_states(mixture,meta['sigma_max'],meta['particles'],meta['seeds'],meta['prior'])
    samples=sample_coupled_refinement(field,meta['gammas'],meta['lambdas'],initial,meta['seeds'],coarse_clock,
                                      integrator=meta.get('sampler_integrator','heun'))
    stem=f"{record['run']}_epoch{record['epoch']:02d}"
    with np.load(out/'checkpoints'/f'{stem}.npz') as f:
        coarse={key:f[key] for key in f.files}
    base=meta['gammas'].index(0.);full=meta['lambdas'].index(1.);affine=meta['lambdas'].index(0.)
    bins=64 if 64 in meta['bins'] else meta['bins'][len(meta['bins'])//2]
    counts=histogram_counts(samples,target_partition(mixture,meta['sigma_min'],bins))
    refined_kl=binned_kl(counts.sum(axis=2),meta['pseudocount'])
    coarse_kl=binned_kl(coarse[f'counts_{bins}'].sum(axis=2),meta['pseudocount'])
    fine_ratio=np.log10(refined_kl/refined_kl[base]);coarse_ratio=np.log10(coarse_kl/coarse_kl[base])
    fine_effect=fine_ratio[:,full]-fine_ratio[:,affine]
    coarse_effect=coarse_ratio[:,full]-coarse_ratio[:,affine]
    # Paired delete-one-seed SE of the coarse-grid effect, for scale context.
    cc=coarse[f'counts_{bins}'];nseeds=len(meta['seeds'])
    loo=binned_kl(cc.sum(axis=2)[:,:,None,:]-cc,meta['pseudocount'])
    loo_ratio=np.log10(loo/loo[base]);loo_effect=loo_ratio[:,full]-loo_ratio[:,affine]
    se=np.sqrt((nseeds-1)/nseeds*np.sum((loo_effect-loo_effect.mean(axis=1,keepdims=True))**2,axis=1))
    change=abs(fine_effect-coarse_effect)
    # A 0.03 dex absolute budget, with two MC SEs where the estimate is noisy.
    tolerance=np.maximum(.03,2*se)
    result=dict(run=record['run'],epoch=record['epoch'],bins=bins,
                max_absolute_kl_change=float(np.max(abs(refined_kl-coarse_kl))),
                max_absolute_log_ratio_change=float(np.max(abs(fine_ratio-coarse_ratio))),
                max_absolute_effect_change=float(np.max(change)),
                max_effect_change_over_tolerance=float(np.max(change/tolerance)),
                passed=bool(np.all(change<=tolerance)),
                table_relative_rms_error_max=table_validation['relative_rms_error_max'],
                coarse_effect=coarse_effect.tolist(),refined_effect=fine_effect.tolist(),
                coarse_effect_mc_se=se.tolist())
    (out/'validation').mkdir(exist_ok=True)
    np.savez_compressed(out/'validation'/f'{stem}.npz',counts=counts,coarse_kl=coarse_kl,refined_kl=refined_kl,
                        coarse_effect=coarse_effect,refined_effect=fine_effect,coarse_effect_mc_se=se)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_ABLATION_OUTPUT)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--checkpoint-ids',nargs='+',default=['default3:0','default3:24','default3:47',
                                                       'default4:0','default4:24','default4:47',
                                                       'default5:0','default5:24','default5:47'])
    args=p.parse_args();out=args.output_dir.resolve()
    meta=json.loads((out/'metadata.json').read_text())
    source=Path(meta['root'])/'paper_odds/outputs/score_error_modes/checkpoint_summary.csv'
    with source.open() as f:
        records=[dict(run=r['run'],epoch=int(r['epoch']),checkpoint=r['checkpoint']) for r in csv.DictReader(f)
                 if f"{r['run']}:{int(r['epoch'])}" in args.checkpoint_ids]
    if len(records)!=len(set(args.checkpoint_ids)) or any(not (out/'checkpoints'/f"{r['run']}_epoch{r['epoch']:02d}.npz").exists() for r in records):
        raise ValueError('Compute all requested coarse checkpoints before validating')
    checks=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(validate_one,r,meta,str(out)) for r in records]
        for future in as_completed(futures):
            check=future.result();checks.append(check)
            print(f"{check['run']} epoch {check['epoch']}: effect change {check['max_absolute_effect_change']:.4g} dex; passed={check['passed']}",flush=True)
    checks.sort(key=lambda r:(r['run'],r['epoch']))
    result=dict(primary_fingerprint=meta['fingerprint'],passed=all(c['passed'] for c in checks),
                sampler_integrator=meta.get('sampler_integrator','heun'),
                coupling='Brownian bridge: two fine increments sum to each original coarse increment',
                refined_steps=2*meta['steps'],refined_table_nodes=2*meta['table_nodes']-1,
                refined_quadrature_order=2*meta['quadrature_order'],
                criterion='Effect change <= max(0.03 dex, 2 paired MC standard errors) at every gamma/direction',
                checks=checks)
    (out/'resolution_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:
        raise AssertionError('Some ablation effects are resolution-sensitive; inspect resolution_validation.json')
    print('Paired refinement validation passed.',flush=True)


if __name__=='__main__':
    main()
