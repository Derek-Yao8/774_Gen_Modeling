"""Load recorded splits without refitting preprocessing or touching test values by default."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy import sparse
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data import load_dataset
from .models import build_model
from .planning import require
from .training_data import verify_source, subset
from .evaluation_metrics import image_features, fit_classifier
from .image_assessment import fit_features


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def indices(a, size):
    require(a.ndim==1 and a.dtype.kind in 'iu' and len(a)>0 and len(np.unique(a))==len(a), 'Invalid or duplicated split indices')
    require(a.min()>=0 and a.max()<size,'Split index out of bounds')
    return a


def manifest(run):
    paths=['execution.json','effective_plan.json','source_modeling_plan.json','input_profile.json']
    names=[m['name'] for m in read(run/'effective_plan.json')['plan']['models']]
    for name in names:
        paths += [f'{name}/{f}' for f in ('job.json','result.json','preprocessing.json','splits.npz','best.pt','generated.npz','history.json')]
    return {p:digest(run/p) if (run/p).is_file() else None for p in paths}


def load_training(run, name, options, dataset=None):
    folder=run/name
    result, job, meta = [read(folder/f) for f in ('result.json','job.json','preprocessing.json')]
    require(result['status'] in ('trained','pilot_trained'),'Training did not complete successfully')
    config=job['config']
    require(config['name']==name,'Job/model mismatch')
    source=Path(dataset or job['dataset'])
    profile=read(run/'input_profile.json')
    require(job['plan_sha256']==digest(run/'source_modeling_plan.json'),'Job does not match saved plan')
    checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    require(checkpoint['plan_sha256']==job['plan_sha256'] and checkpoint['config']==config,'Checkpoint/plan mismatch')
    require(checkpoint['preprocessing']==meta,'Checkpoint/preprocessing mismatch')
    split_file=np.load(folder/'splits.npz',allow_pickle=False)
    splits={k:split_file[k] for k in split_file.files}
    data=load_dataset(source,split='train' if meta['kind']=='image' else None)
    verify_source(data,profile)
    train_i=indices(splits['train_indices'],data.X.shape[0])
    if meta['kind'] in ('counts','continuous'):
        vi,ti=[indices(splits[k],data.X.shape[0]) for k in ('validation_indices','test_indices')]
        require(not(set(train_i)&set(vi) or set(train_i)&set(ti) or set(vi)&set(ti)),'Train/validation/test overlap')
        features=np.asarray(meta['selected_feature_positions'])
        indices(features,data.X.shape[1])
        require([str(data.var.index[i]) for i in features]==meta['selected_feature_ids'],'Selected gene IDs no longer match input')
    else:
        require(tuple(data.images.shape[1:])==tuple(meta['shape']),'Image shape changed')
    raw_y=None if data.labels is None else data.labels.iloc[:,0].to_numpy()
    train_i=subset(train_i,raw_y,options['max_train'],options['seed'])
    if meta['kind']=='image':
        train=data.images[train_i]
    else:
        x=data.X[train_i][:,features]
        train=x.toarray() if sparse.issparse(x) else np.asarray(x)
    if meta['kind']=='continuous':
        from .flows import log_dequantize
        train=log_dequantize(train,train_i,meta['dequantization_seed'])
    y=None if raw_y is None else raw_y[train_i]
    with np.load(folder/'generated.npz',allow_pickle=False) as generated:
        payload={k:generated[k] for k in generated.files if k != 'decoder_means'}
    return dict(run=run,folder=folder,source=source,profile=profile,job=job,config=config,meta=meta,
                checkpoint=checkpoint,splits=splits,train=train,train_y=y,train_indices=train_i,generated=payload,
                task=job['plan']['task'])


def load_holdout(bundle, mode, options):
    meta, splits, source=bundle['meta'],bundle['splits'],bundle['source']
    if meta['kind']=='image':
        data=load_dataset(source,split='val' if mode=='validation' else 'test')
        require(tuple(data.images.shape[1:])==tuple(meta['shape']),'Held-out image shape mismatch')
        ix=indices(splits['validation_indices'],data.X.shape[0]) if mode=='validation' else np.arange(data.X.shape[0])
        y=data.labels.iloc[:,0].to_numpy()
        ix=subset(ix,y,options['max_holdout'],options['seed']+1)
        return data.images[ix],y[ix],ix
    data=load_dataset(source)
    verify_source(data,bundle['profile'])
    key='validation_indices' if mode=='validation' else 'test_indices'
    ix=indices(splits[key],data.X.shape[0])
    ix=subset(ix,None,options['max_holdout'],options['seed']+1)
    x=data.X[ix][:,meta['selected_feature_positions']]
    x=x.toarray() if sparse.issparse(x) else np.asarray(x)
    if meta['kind']=='continuous':
        from .flows import log_dequantize
        x=log_dequantize(x,ix,meta['dequantization_seed'])
    return x,None,ix


def encode(bundle, x):
    checkpoint=bundle['checkpoint']
    require(not bundle['config']['conditional'],'Conditional embeddings would require held-out labels; representation evaluation requires an unconditional encoder')
    model=build_model(checkpoint['config'],checkpoint['shape'],0)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    vectors=[]
    torch.set_num_threads(2)
    with torch.no_grad():
        for start in range(0,len(x),128):
            batch=np.asarray(x[start:start+128],dtype=np.float32)
            if bundle['meta']['kind']=='image':
                if batch.ndim==3: batch=batch[...,None]
                batch=batch.transpose(0,3,1,2)/255
            mu,_=model.encode(torch.from_numpy(np.ascontiguousarray(batch)))
            vectors.append(mu.numpy())
    return np.concatenate(vectors)


def fit_evaluators(bundle, options):
    train, y, kind=bundle['train'],bundle['train_y'],bundle['meta']['kind']
    feature=image_features(train) if kind=='image' else train if kind=='continuous' else np.log1p(train)
    scaler=StandardScaler().fit(feature)
    fit={'scaler':scaler,'task':bundle['task'],'converged':True,'utility_status':'not_applicable'}
    if kind=='image' and len(bundle['generated']['samples']):
        fit['image_pca'], fit['texture_scaler'], fit['pixel_pca_scaler'] = fit_features(train, options['seed'])
    if bundle['task']=='class_balancing':
        require(y is not None and len(np.unique(y))>=2,'Augmentation evaluation needs training labels')
        generated=bundle['generated']
        require('labels' in generated,'Synthetic labels are required for augmentation')
        x=scaler.transform(feature)
        g=scaler.transform(image_features(generated['samples']))
        gy=generated['labels']
        rng=np.random.default_rng(options['seed'])
        control=np.array([rng.choice(np.flatnonzero(y==c)) for c in gy])
        fit['baseline'],c1=fit_classifier(x,y)
        fit['alternative'],c2=fit_classifier(np.concatenate([x,g]),np.concatenate([y,gy]))
        fit['resampling_control'],c3=fit_classifier(np.concatenate([x,x[control]]),np.concatenate([y,y[control]]))
        fit.update(converged=c1 and c2 and c3,utility_status='computed',classes=np.unique(y))
    elif bundle['task']=='representation_learning':
        if y is None:
            fit['utility_status']='unavailable_without_labels'
            return fit
        raw=train.reshape(len(train),-1).astype(np.float32)/255 if kind=='image' else np.log1p(train)
        dim=min(bundle['config']['latent_dim'],raw.shape[1],len(raw)-1)
        fit['pca']=PCA(n_components=dim,svd_solver='randomized',random_state=options['seed']).fit(raw)
        z=encode(bundle,train)
        fit['embedding_scaler']=StandardScaler().fit(z)
        fit['pca_scaler']=StandardScaler().fit(fit['pca'].transform(raw))
        fit['baseline'],c1=fit_classifier(fit['pca_scaler'].transform(fit['pca'].transform(raw)),y)
        fit['alternative'],c2=fit_classifier(fit['embedding_scaler'].transform(z),y)
        fit.update(classes=np.unique(y),converged=c1 and c2,utility_status='computed')
    return fit


def fitted_signature(fit):
    h=hashlib.sha256()
    for key,value in sorted(fit.items()):
        h.update(key.encode())
        for attr in ('mean_','scale_','components_','coef_','intercept_','classes_','explained_variance_ratio_'):
            if hasattr(value,attr):
                h.update(attr.encode())
                h.update(np.ascontiguousarray(getattr(value,attr)).tobytes())
    return h.hexdigest()
