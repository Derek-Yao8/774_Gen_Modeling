"""Alternating non-saturating GAN training and validation-only checkpoint selection."""
import math
import os
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

from .capabilities import TRAINERS
from .models import build_model
from .planning import require, save
from .training_data import prepare, subset
from .evaluation_metrics import image_features
from .trainer import seed_all, check_limits, choose_device, tensors, save_checkpoint, ResourceLimit


def step(model, x, y, generator_optimizer, discriminator_optimizer):
    discriminator_optimizer.zero_grad(set_to_none=True)
    generator_optimizer.zero_grad(set_to_none=True)
    z=torch.randn(len(x),model.latent_dim,device=x.device)
    d=model.discriminator_loss(x,z,y)
    require(torch.isfinite(d).item(), 'Nonfinite discriminator loss')
    d.backward()
    torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(),10,error_if_nonfinite=True)
    discriminator_optimizer.step()
    discriminator_optimizer.zero_grad(set_to_none=True)
    for p in model.discriminator.parameters(): p.requires_grad_(False)
    try:
        g=model.generator_loss(torch.randn_like(z),y)
        require(torch.isfinite(g).item(), 'Nonfinite generator loss')
        g.backward()
        torch.nn.utils.clip_grad_norm_(model.generator.parameters(),10,error_if_nonfinite=True)
        generator_optimizer.step()
    finally:
        for p in model.discriminator.parameters(): p.requires_grad_(True)
    return float(d.item()),float(g.item())


def draws(model,z,y,shape,batch_size,device,deadline,max_bytes):
    output=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(z),batch_size):
            check_limits(deadline,max_bytes,device)
            labels=None if y is None else torch.from_numpy(y[start:start+batch_size]).to(device)
            pixels=model.decode(torch.from_numpy(z[start:start+batch_size]).to(device),labels)
            require(torch.isfinite(pixels).all().item(),'Nonfinite generated pixels')
            pixels=np.rint(np.clip((pixels.cpu().numpy().transpose(0,2,3,1)+1)/2,0,1)*255).astype(np.uint8)
            output.append(pixels[...,0] if len(shape)==2 else pixels)
    return np.concatenate(output) if output else np.empty((0,*shape),dtype=np.uint8)


def score_samples(scaler, real, samples, labels):
    real=scaler.transform(image_features(real))
    fake=scaler.transform(image_features(samples))
    def discrepancy(a,b):
        return float(np.mean([wasserstein_distance(a[:,j],b[:,j]) for j in range(a.shape[1])]))
    scores=[discrepancy(real,fake)]
    if labels is not None:
        scores += [discrepancy(real[labels==c],fake[labels==c]) for c in np.unique(labels)]
    return float(np.mean(scores))


def plot(out,history,samples):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    epochs=[r['epoch'] for r in history]
    for key in ('discriminator_loss','generator_loss'):
        axes[0].plot(epochs,[r[key] for r in history],label=key)
    axes[0].legend(); axes[0].set_title('Adversarial losses (not quality scores)')
    axes[1].plot(epochs,[r['validation_feature_wasserstein'] for r in history])
    axes[1].set_title('Validation checkpoint score (lower is closer)')
    for ax in axes: ax.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(out/'learning_curve.png',dpi=140);plt.close(fig)
    if len(samples):
        fig,axes=plt.subplots(2,8,figsize=(12,3))
        for i,ax in enumerate(axes.flat):
            ax.axis('off')
            if i<len(samples): ax.imshow(samples[i].squeeze(),cmap='gray',vmin=0,vmax=255)
        fig.suptitle('GAN generated samples — no added observation noise')
        fig.tight_layout();fig.savefig(out/'generated_grid.png',dpi=140);plt.close(fig)


def train_job(job,out):
    started=time.monotonic()
    config,plan=job['config'],job['plan']
    require(plan['task'] in ('data_generation','class_balancing'),'GAN does not provide representations')
    seed=plan['split']['seed']; max_bytes=job['max_bytes']
    deadline=started+job['wall_seconds']; train_deadline=started+.8*job['wall_seconds']
    torch.set_num_threads(min(4,os.cpu_count() or 1));seed_all(seed)
    data=prepare(job['dataset'],job['profile'],plan,config,out,job['pilot'],max_bytes)
    check_limits(deadline,max_bytes)
    model=build_model(config,data.shape,len(data.classes))
    device,probes=choose_device(model,data,job['device'])
    save(out/'device.json',{'requested':job['device'],'selected':device,'probes':probes})
    seed_all(seed);model=model.to(device)
    gopt=torch.optim.Adam(model.generator.parameters(),lr=config['learning_rate'],betas=(.5,.999))
    dopt=torch.optim.Adam(model.discriminator.parameters(),lr=config['learning_rate'],betas=(.5,.999))
    ti=subset(np.arange(len(data.train)),data.train_labels,512,seed+10)
    vi=subset(np.arange(len(data.val)),data.val_labels,512,seed+11)
    scaler=StandardScaler().fit(image_features(data.train[ti]))
    labels=None if data.val_labels is None else data.val_labels[vi]
    fixed_z=np.random.default_rng(seed+12).standard_normal((len(vi),model.latent_dim)).astype(np.float32)
    np.savez_compressed(out/'checkpoint_selection.npz',training_positions=ti,validation_positions=vi,
                        feature_mean=scaler.mean_,feature_scale=scaler.scale_,latents=fixed_z,
                        labels=np.array([],dtype=np.int64) if labels is None else labels)
    history=[];best=float('inf');best_epoch=None;stale=0;stop='epochs_completed'
    rng=np.random.default_rng(seed)
    for epoch in range(1,config['epochs']+1):
        model.train(); totals=np.zeros(2); seen=0
        try:
            order=rng.permutation(len(data.train))
            for start in range(0,len(order),config['batch_size']):
                check_limits(train_deadline,max_bytes,device)
                ix=order[start:start+config['batch_size']]
                x,y=tensors(data,ix,device)
                totals+=np.array(step(model,x,y,gopt,dopt))*len(ix);seen+=len(ix)
            generated=draws(model,fixed_z,labels,data.shape,config['batch_size'],device,train_deadline,max_bytes)
            score=score_samples(scaler,data.val[vi],generated,labels)
            require(math.isfinite(score),'Nonfinite checkpoint score')
            check_limits(train_deadline,max_bytes,device)
        except ResourceLimit as exc:
            stop=str(exc);break
        row={'epoch':epoch,'discriminator_loss':float(totals[0]/seen),'generator_loss':float(totals[1]/seen),
             'validation_feature_wasserstein':score,'elapsed_seconds':time.monotonic()-started}
        history.append(row);save(out/'history.json',history)
        optimizers={'generator':gopt.state_dict(),'discriminator':dopt.state_dict()}
        if score<best:
            best=score;best_epoch=epoch;stale=0
            save_checkpoint(out/'best.pt',model,optimizers,config,data,epoch,seed,job['plan_sha256'])
        else: stale+=1
        save_checkpoint(out/'last.pt',model,optimizers,config,data,epoch,seed,job['plan_sha256'])
        print(f'image_gan epoch {epoch}: D={row["discriminator_loss"]:.4f}, G={row["generator_loss"]:.4f}, validation={score:.4f}',flush=True)
        if stale>=config['patience']: stop='early_stopping';break
    require(best_epoch is not None,'No full GAN epoch/validation completed within resource limits')
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    n=config['generate_samples']
    require(n*(math.prod(data.shape)*8+model.latent_dim*8)<max_bytes//2,'Generated arrays exceed memory allowance')
    z=np.random.default_rng(seed+2000).standard_normal((n,model.latent_dim)).astype(np.float32)
    y=np.arange(n,dtype=np.int64)%len(data.classes) if data.classes else None
    samples=draws(model,z,y,data.shape,config['batch_size'],device,deadline,max_bytes)
    output={'samples':samples}
    if y is not None: output.update(label_indices=y,labels=np.asarray(data.classes)[y])
    np.savez_compressed(out/'generated.npz',**output)
    check_limits(deadline,max_bytes,device);plot(out,history,samples)
    result={'status':'pilot_trained' if job['pilot'] else 'trained','model':'image_gan','device':device,
            'best_epoch':best_epoch,'best_validation_feature_wasserstein':best,'epochs_completed':len(history),
            'stop_reason':stop,'generated_samples':n,'elapsed_seconds':time.monotonic()-started,
            'train_observations':len(data.train),'validation_observations':len(data.val),
            'test_used':False,'quality_evaluation_complete':False,'trainer_contract':TRAINERS['image_gan'],
            'limitations':['Adversarial losses and pooled-feature checkpoint selection do not establish fidelity or downstream utility.',
                           'Validation is reused for checkpoint selection; final testing remains reserved.',
                           'GAN instability and mode collapse require independent quality/diversity checks.']}
    save(out/'result.json',result)
    return result
