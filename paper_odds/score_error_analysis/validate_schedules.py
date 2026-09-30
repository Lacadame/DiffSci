"""Paired twofold step refinement of representative checkpoint schedules.

Run with the fingerprinted result directory as the sole argument. The fine
Brownian increments sum exactly to each coarse increment. This is a numerical
resolution diagnostic, not an uncertainty interval over trained networks.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np

from .schedule_experiment import build_field, sample_schedules, schedule_matrix, write_csv
from .residual_ablation import (AblationField, brownian_generators, shared_initial_states,
                               target_partition, histogram_counts, binned_kl)
from .score_error_modes import GaussianMixture1D


def main(directory):
    out=Path(directory)
    metadata=json.loads((out/'metadata.json').read_text())
    config=metadata['configuration']
    mixture=GaussianMixture1D(**config['mixture'])
    with np.load(out/'clock.npz') as saved:
        coarse={k:saved[k].copy() for k in saved.files}
    ell=coarse['ell']
    fine_ell=np.sort(np.r_[ell,(ell[:-1]+ell[1:])/2])
    fine_variance=coarse['variance'][0]*np.exp(-fine_ell)
    fine_sigma=np.sqrt(np.maximum(fine_variance-mixture.variance, config['sigma_min']**2))
    fine_sigma[[0,-1]]=[config['sigma_max'],config['sigma_min']]
    fine=dict(ell=fine_ell,sigma=fine_sigma,variance=mixture.variance+fine_sigma**2,
              sqrt_variance=np.sqrt(mixture.variance+fine_sigma**2))
    seeds=config['seeds'];particles=config['particles']
    initial=shared_initial_states(mixture,config['sigma_max'],particles,seeds,'exact')
    generators=brownian_generators(seeds)
    bridge=[np.random.default_rng(np.random.SeedSequence([seed,7210])) for seed in seeds]
    increments=np.empty((len(ell)-1,len(seeds),particles))
    fine_increments=np.empty((2*(len(ell)-1),len(seeds),particles))
    for i,h in enumerate(np.diff(ell)):
        dw=np.stack([rng.standard_normal(particles) for rng in generators])*np.sqrt(h)
        split=np.stack([rng.standard_normal(particles) for rng in bridge])*np.sqrt(h)/2
        increments[i]=dw
        fine_increments[2*i]=dw/2+split
        fine_increments[2*i+1]=dw/2-split
    np.testing.assert_allclose(fine_increments[::2]+fine_increments[1::2],increments,atol=1e-15,rtol=1e-15)
    wanted={('default3',0),('default4',24),('default5',47)}
    records=[r for r in config['checkpoints'] if (r['run'],r['epoch']) in wanted]
    if not records:
        records=config['checkpoints'][:1]
    keys=['ode','c0.2','c1','c5','mid5','burst50','pop5','tail5']
    rows=[];start=time.time()
    for record in records:
        cp=f"{record['run']}:{record['epoch']}"
        specs=[]
        active_keys=[]
        for key in keys:
            if key not in config['schedules'][cp]:continue
            spec=config['schedules'][cp][key].copy()
            for bound in ['lo','hi']:
                if bound in spec:spec[bound]=float(spec[bound])
            specs.append(spec);active_keys.append(key)
        field,validation=build_field(record,config,fine,'cpu')
        coarse_field=AblationField(mixture,coarse,field.b[::2],field.C[::2],field.learned_table[::2],field.model,field.z_limit)
        counts=[]
        # Exact control, affine error and full score all use the same noise.
        for label,current_clock,current_field,noise in [('coarse',coarse,coarse_field,increments),('fine',fine,field,fine_increments)]:
            samples=sample_schedules(current_field,schedule_matrix(current_clock,mixture,specs),
                                     [None,0.,1.],initial,seeds,noise_increments=noise)
            counts.append(histogram_counts(samples,target_partition(mixture,config['sigma_min'],64)))
        ck=np.stack(counts) # resolution, schedule, arm, seed, bin
        pooled=binned_kl(ck.sum(axis=3),config['pseudocount'])
        lr=np.log10(pooled/pooled[:,:1])
        leave=binned_kl(ck.sum(axis=3,keepdims=True)-ck,config['pseudocount'])
        leave_lr=np.log10(leave/leave[:,:1])
        diff=leave_lr[1]-leave_lr[0]
        se=np.sqrt((len(seeds)-1)/len(seeds)*np.sum((diff-diff.mean(axis=2,keepdims=True))**2,axis=2))
        for j,key in enumerate(active_keys):
            for k,arm in enumerate(['exact','affine','full']):
                for q,direction in enumerate(['q_p','p_q']):
                    rows.append(dict(run=record['run'],epoch=record['epoch'],schedule=key,arm=arm,direction=direction,
                        coarse_steps=len(ell)-1,fine_steps=len(fine_ell)-1,
                        coarse_kl=float(pooled[0,j,k,q]),fine_kl=float(pooled[1,j,k,q]),
                        coarse_log_ratio=float(lr[0,j,k,q]),fine_log_ratio=float(lr[1,j,k,q]),
                        log_ratio_change=float(lr[1,j,k,q]-lr[0,j,k,q]),paired_delete_seed_SE=float(se[j,k,q]),
                        above_floor=bool(np.all(pooled[:,[0,j],k,q] > 3*pooled[:,[0,j],0,q])) if arm!='exact' else False))
        write_csv(out/'resolution_validation.csv',rows)
        print(cp,'finished',round(time.time()-start,1),'seconds',flush=True)
    (out/'resolution_validation.json').write_text(json.dumps(dict(
        fingerprint=metadata['fingerprint'],checkpoints=[f"{r['run']}:{r['epoch']}" for r in records],
        coarse_steps=len(ell)-1,fine_steps=len(fine_ell)-1,bins=64,seeds=seeds,particles=particles,
        coupling='common coarse increments; fine Brownian bridge increments sum to coarse',
        seconds=time.time()-start),indent=2)+'\n')


if __name__=='__main__':main(sys.argv[1])
