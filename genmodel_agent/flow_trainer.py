"""Maximum likelihood training of standalone expression flows."""
import os
import time
import numpy as np
import torch
from scipy import sparse
from .flows import log_dequantize
from .models import build_model
from .capabilities import TRAINERS
from .planning import require,save
from .training_data import prepare
from .trainer import (seed_all,check_limits,choose_device,tensors,validate_epoch,
                      save_checkpoint,ResourceLimit)


def prepare_flow(job,out):
    d=prepare(job['dataset'],job['profile'],job['plan'],job['config'],out,job['pilot'],job['max_bytes'])
    seed=job['plan']['split']['seed']+4000
    with np.load(out/'splits.npz') as splits:
        train=log_dequantize(d.train.toarray() if sparse.issparse(d.train) else d.train,splits['train_indices'],seed)
        val=log_dequantize(d.val.toarray() if sparse.issparse(d.val) else d.val,splits['validation_indices'],seed)
    mean=train.mean(0);scale=np.maximum(train.std(0),1e-6)
    d.train=((train-mean)/scale).astype(np.float32);d.val=((val-mean)/scale).astype(np.float32)
    d.metadata.update(kind='continuous',output_domain='log_dequantized_expression',
        dequantization_seed=seed,flow_mean=mean.tolist(),flow_scale=scale.tolist(),
        flow_preprocessing='log(raw selected count + fixed per-cell uniform jitter [1e-6,1)); train mean/std only',
        density_domain='standardized continuous log-dequantized expression; not raw counts',library_normalization=False)
    save(out/'preprocessing.json',d.metadata)
    return d


def train_job(job,out):
    started=time.monotonic();c=job['config'];seed=job['plan']['split']['seed']
    limit=job['max_bytes'];deadline=started+job['wall_seconds'];training_deadline=started+.8*job['wall_seconds']
    torch.set_num_threads(min(4,os.cpu_count() or 1));seed_all(seed)
    d=prepare_flow(job,out)
    model=build_model(c,d.shape,0)
    device,probes=choose_device(model,d,job['device'])
    save(out/'device.json',{'requested':job['device'],'selected':device,'probes':probes})
    seed_all(seed);model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    history=[];best=float('inf');best_epoch=None;stale=0;stop='epochs_completed'
    rng=np.random.default_rng(seed)
    for epoch in range(1,c['epochs']+1):
        model.train();total=0.;seen=0
        try:
            order=rng.permutation(len(d.train))
            for start in range(0,len(order),c['batch_size']):
                check_limits(training_deadline,limit,device)
                ix=order[start:start+c['batch_size']];x,_=tensors(d,ix,device)
                optimizer.zero_grad(set_to_none=True);loss=model.loss(x)[0]
                require(torch.isfinite(loss).item(),'Nonfinite flow NLL')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step()
                total+=loss.item()*len(ix);seen+=len(ix)
            val=validate_epoch(model,d,c['batch_size'],device,training_deadline,limit,seed)[0]
        except ResourceLimit as exc:
            stop=str(exc);break
        row={'epoch':epoch,'train_nll':total/seen,'validation_nll':val,'elapsed_seconds':time.monotonic()-started}
        history.append(row);save(out/'history.json',history)
        if val<best:
            best=val;best_epoch=epoch;stale=0
            save_checkpoint(out/'best.pt',model,optimizer,c,d,epoch,seed,job['plan_sha256'])
        else:stale+=1
        save_checkpoint(out/'last.pt',model,optimizer,c,d,epoch,seed,job['plan_sha256'])
        print(f'{c["name"]} epoch {epoch}: NLL={total/seen:.3f}, validation={val:.3f}',flush=True)
        if stale>=c['patience']:stop='early_stopping';break
    require(best_epoch is not None,'No complete flow epoch within resource limits')
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True)['state_dict']);model.eval()
    n=c['generate_samples'];rows=[]
    require(n*c['selected_features']*32<limit//2,'Generated flow arrays exceed allowance')
    rng=np.random.default_rng(seed+2000)
    with torch.no_grad():
        for start in range(0,n,c['batch_size']):
            check_limits(deadline,limit,device)
            z=torch.from_numpy(rng.standard_normal((min(c['batch_size'],n-start),c['latent_dim'])).astype(np.float32)).to(device)
            x=model.decode(z)
            recovered,_=model(x)
            require(torch.allclose(recovered,z,atol=1e-3,rtol=1e-3),'Flow inverse consistency failed')
            sample=x.cpu().numpy()*np.asarray(d.metadata['flow_scale'])+np.asarray(d.metadata['flow_mean'])
            require(np.isfinite(sample).all(),'Nonfinite flow samples')
            rows.append(sample)
    samples=np.concatenate(rows)
    np.savez_compressed(out/'generated.npz',samples=samples,
        selected_feature_ids=np.asarray(d.metadata['selected_feature_ids']),output_domain='log_dequantized_expression')
    check_limits(deadline,limit,device)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for key in ('train_nll','validation_nll'):ax.plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    ax.set(xlabel='Epoch',ylabel='NLL in standardized continuous coordinates',title=c['name']);ax.legend()
    fig.tight_layout();fig.savefig(out/'learning_curve.png',dpi=140);plt.close(fig)
    result={'status':'pilot_trained' if job['pilot'] else 'trained','model':c['name'],'device':device,
        'best_epoch':best_epoch,'best_validation_nll':best,'epochs_completed':len(history),'stop_reason':stop,
        'generated_samples':n,'elapsed_seconds':time.monotonic()-started,'train_observations':len(d.train),
        'validation_observations':len(d.val),'test_used':False,'quality_evaluation_complete':False,
        'trainer_contract':TRAINERS[c['name']],
        'limitations':['Samples are continuous log-dequantized selected-gene expression, not integer counts.',
                       'NLL is in standardized coordinates, not a discrete count likelihood or ELBO.',
                       'Planar/radial expressiveness is limited; no biological/downstream benefit is established.']}
    save(out/'result.json',result)
    return result
