"""Measured diagnostics with real references; all learned transforms fit training only."""
import hashlib
import warnings

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def image_features(x):
    x = np.asarray(x, dtype=np.float64) / 255
    if x.ndim == 3:
        x = x[..., None]
    h, w, c = x.shape[1:]
    pooled = x.reshape(len(x), 4, h//4, 4, w//4, c).mean(axis=(2,4)).reshape(len(x), -1)
    return np.column_stack([pooled, x.std(axis=(1,2))])


def row_hashes(x):
    # Canonical numeric encoding lets int32/int64 counts compare by value.
    return [hashlib.sha256(np.ascontiguousarray(row, dtype='<f8').tobytes()).hexdigest() for row in x]


def safe_ratio(a, b):
    return float(a/b) if b > 1e-12 else (1.0 if a <= 1e-12 else None)


def validity(samples, shape, kind, expected_count, labels=None, expected_classes=None, feature_ids=None, expected_ids=None):
    errors = []
    if samples.shape != (expected_count, *shape): errors.append('Generated shape/count does not match the executed plan')
    numeric = samples.dtype.kind in 'biuf'
    if not numeric or not np.isfinite(samples).all(): errors.append('Samples are not finite real numeric values')
    if numeric and samples.size:
        if kind == 'image' and (samples.dtype != np.uint8 or samples.min() < 0 or samples.max() > 255):
            errors.append('Image encoding must be uint8 pixels')
        if kind == 'counts' and ((samples < 0).any() or not np.equal(samples, np.floor(samples)).all()):
            errors.append('Counts must be nonnegative integers')
    if expected_classes:
        if labels is None or labels.shape != (len(samples),) or not np.isin(labels, expected_classes).all():
            errors.append('Generated labels are missing, misaligned or outside the fitted class vocabulary')
    if expected_ids is not None and (feature_ids is None or list(feature_ids) != list(expected_ids)):
        errors.append('Generated gene IDs do not match the fitted selected-gene panel')
    return {'passed': not errors, 'errors': errors, 'observations': len(samples), 'shape': list(samples.shape), 'dtype': str(samples.dtype)}


def distribution_discrepancies(x, y, kind, feature_columns=None):
    if kind == 'image':
        return {'feature_wasserstein': float(np.mean([wasserstein_distance(x[:,j], y[:,j]) for j in range(x.shape[1])]))}
    if kind == 'continuous':
        def corr(a):
            a=a[:,feature_columns]-a[:,feature_columns].mean(0)
            norms=np.linalg.norm(a,axis=0)
            return (a.T@a)/np.maximum(norms[:,None]*norms[None,:],1e-12)
        ix=np.triu_indices(len(feature_columns),1)
        return {'continuous_feature_wasserstein':float(np.mean([wasserstein_distance(x[:,j],y[:,j]) for j in range(x.shape[1])])),
                'continuous_correlation_mae':float(np.abs(corr(x)[ix]-corr(y)[ix]).mean())}
    result = {
        'log_gene_mean_mae': float(np.abs(np.log1p(x.mean(0))-np.log1p(y.mean(0))).mean()),
        'log_gene_variance_mae': float(np.abs(np.log1p(x.var(0))-np.log1p(y.var(0))).mean()),
        'gene_zero_fraction_mae': float(np.abs((x==0).mean(0)-(y==0).mean(0)).mean()),
        'log_panel_total_wasserstein': float(wasserstein_distance(np.log1p(x.sum(1)), np.log1p(y.sum(1))))}
    def correlation(a):
        a = np.log1p(a[:,feature_columns])
        a = a-a.mean(0)
        lengths = np.linalg.norm(a,axis=0)
        return (a.T@a)/np.maximum(lengths[:,None]*lengths[None,:], 1e-12)
    ix = np.triu_indices(len(feature_columns),1)
    result['gene_correlation_mae'] = float(np.abs(correlation(x)[ix]-correlation(y)[ix]).mean()) if len(ix[0]) else None
    return result


def quality(train, holdout, samples, kind, scaler, options, train_labels=None, holdout_labels=None, sample_labels=None):
    rng = np.random.default_rng(options['seed'])
    if not len(samples):
        return {'status':'not_applicable_no_generation','quality_screen_passed':False}, {}
    raw_t, raw_h, raw_s = train, holdout, samples
    feature = image_features if kind == 'image' else (lambda a: a) if kind == 'continuous' else np.log1p
    t, h, s = [scaler.transform(feature(a)) for a in (train,holdout,samples)]
    k = min(len(samples), len(holdout), options['metric_rows'])
    if k < 3:
        return {'status':'insufficient_samples','quality_screen_passed':False}, {}
    si = rng.choice(len(samples), k, replace=False)
    hi = rng.choice(len(holdout), k, replace=False)
    columns = np.argsort(np.var(np.log1p(train),axis=0))[-min(64,train.shape[1]):] if kind == 'counts' else np.argsort(np.var(train,axis=0))[-min(64,train.shape[1]):] if kind == 'continuous' else None
    a, b = (s[si],h[hi]) if kind in ('image','continuous') else (samples[si],holdout[hi])
    discrepancies = distribution_discrepancies(a,b,kind,columns)
    baselines = []
    for _ in range(options['reference_repeats']):
        ri = rng.choice(len(train), k, replace=True)
        ref = t[ri] if kind in ('image','continuous') else train[ri]
        baselines.append(distribution_discrepancies(ref,b,kind,columns))
    fidelity = {}
    for name, value in discrepancies.items():
        vals = [row[name] for row in baselines if row[name] is not None]
        median = float(np.median(vals)) if vals else None
        fidelity[name] = {'synthetic_discrepancy':value,'real_reference_median':median,
                         'reference_range':[float(min(vals)),float(max(vals))] if vals else None,
                         'ratio':safe_ratio(value,median) if value is not None and median is not None else None,
                         'screen_passed': bool(value is not None and median is not None and value <= 1.1*median+1e-8)}
    uncertainty={name:[] for name in discrepancies}
    for _ in range(min(options['bootstrap'],200)):
        ia,ib=rng.integers(k,size=k),rng.integers(k,size=k)
        draw=distribution_discrepancies(a[ia],b[ib],kind,columns)
        for name,value in draw.items():
            if value is not None: uncertainty[name].append(value)
    for name,values in uncertainty.items():
        fidelity[name]['synthetic_discrepancy_bootstrap_95_ci']=np.quantile(values,[.025,.975]).tolist() if values else None
        fidelity[name]['uncertainty_note']='Independent row bootstrap of generated and held-out samples; conditional on one fitted model.'
    ds, dh = float(np.median(pdist(s[si]))), float(np.median(pdist(h[hi])))
    ratio = safe_ratio(ds,dh)
    diversity = {'synthetic_median_distance':ds,'real_median_distance':dh,'ratio':ratio,
                 'screen_passed':ratio is not None and 0.9 <= ratio <= 1.1}
    nn = NearestNeighbors(n_neighbors=2,algorithm='brute').fit(t)
    loo = nn.kneighbors(t)[0][:,1]
    threshold = float(np.quantile(loo,0.01))
    near_s = nn.kneighbors(s,n_neighbors=1)[0][:,0]
    near_h = nn.kneighbors(h,n_neighbors=1)[0][:,0]
    hashes = set(row_hashes(raw_t))
    duplicates_s = float(np.mean([v in hashes for v in row_hashes(raw_s)]))
    duplicates_h = float(np.mean([v in hashes for v in row_hashes(raw_h)]))
    near_rate_s, near_rate_h = float(np.mean(near_s <= threshold)), float(np.mean(near_h <= threshold))
    overlap = {'training_loo_percentile_1':threshold,'synthetic_near_fraction':near_rate_s,
               'heldout_near_fraction':near_rate_h,'synthetic_exact_duplicate_fraction':duplicates_s,
               'heldout_exact_duplicate_fraction':duplicates_h,'training_resampling_exact_duplicate_fraction':1.0,
               'screen_passed': bool(duplicates_s <= duplicates_h and near_rate_s <= near_rate_h+0.02),
               'scope':'Compared only with the recorded training reference subset; not a privacy guarantee.'}
    per_class = {}
    if kind == 'image' and sample_labels is not None:
        for label in np.unique(train_labels):
            it, ih, ig = [np.flatnonzero(l == label) for l in (train_labels,holdout_labels,sample_labels)]
            n = min(len(ih),len(ig),options['metric_rows'])
            if n < 3 or len(it) < 2:
                per_class[str(label)] = {'status':'insufficient_samples','n':n}
                continue
            ig, ih = rng.choice(ig,n,replace=False),rng.choice(ih,n,replace=False)
            value = distribution_discrepancies(s[ig],h[ih],'image')['feature_wasserstein']
            refs = [distribution_discrepancies(t[rng.choice(it,n,replace=True)],h[ih],'image')['feature_wasserstein'] for _ in range(options['reference_repeats'])]
            d = safe_ratio(float(np.median(pdist(s[ig]))),float(np.median(pdist(h[ih]))))
            per_class[str(label)] = {'status':'computed','n':n,'synthetic_discrepancy':value,
                                    'reference_median':float(np.median(refs)),'diversity_ratio':d,
                                    'screen_passed':bool(value <= 1.1*np.median(refs)+1e-8 and d is not None and 0.9 <= d <= 1.1)}
    passed = all(v['screen_passed'] for v in fidelity.values()) and diversity['screen_passed'] and overlap['screen_passed']
    if per_class: passed = passed and all(v.get('screen_passed',False) for v in per_class.values())
    return {'status':'computed','matched_sample_count':k,'fidelity':fidelity,'diversity':diversity,'overlap':overlap,
            'per_class':per_class,'quality_screen_passed':bool(passed)}, {'synthetic_nn_distances':near_s,'heldout_nn_distances':near_h}


def fit_classifier(x, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model = LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,tol=1e-5).fit(x,y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def paired_scores(y, baseline, alternative, classes, bootstrap, seed):
    def score(pred):
        return float(f1_score(y,pred,labels=classes,average='macro',zero_division=0))
    base_recall = recall_score(y,baseline,labels=classes,average=None,zero_division=0)
    alt_recall = recall_score(y,alternative,labels=classes,average=None,zero_division=0)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y==c) for c in classes]
    draws=[]
    for _ in range(bootstrap):
        ix=np.concatenate([rng.choice(v,len(v),replace=True) for v in strata if len(v)])
        draws.append(f1_score(y[ix],alternative[ix],labels=classes,average='macro',zero_division=0)-
                     f1_score(y[ix],baseline[ix],labels=classes,average='macro',zero_division=0))
    ci=np.quantile(draws,[0.025,0.975]).tolist()
    delta=score(alternative)-score(baseline)
    return {'baseline_macro_f1':score(baseline),'alternative_macro_f1':score(alternative),'macro_f1_delta':delta,
            'paired_bootstrap_95_ci':ci,'bootstrap_note':'Class-stratified resampling of held-out observations only; excludes training-seed uncertainty.',
            'per_class':[{'label':str(c),'heldout_count':int((y==c).sum()),'baseline_recall':float(a),'alternative_recall':float(b)} for c,a,b in zip(classes,base_recall,alt_recall)],
            'baseline_confusion_matrix':confusion_matrix(y,baseline,labels=classes).tolist(),
            'alternative_confusion_matrix':confusion_matrix(y,alternative,labels=classes).tolist(),
            'utility_screen_passed':bool(all(len(s)>0 for s in strata) and delta >= 0.01 and ci[0]>0 and np.all(alt_recall-base_recall >= -0.02))}
