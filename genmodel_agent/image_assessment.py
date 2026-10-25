"""Supplementary texture/PCA distribution evidence and reproducible visual packets.

Handcrafted gradients are not canonical HOG, FID or a learned histology score.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .planning import save


def texture_features(images):
    a = np.asarray(images, dtype=np.float64) / 255
    if a.ndim == 3:
        a = a[..., None]
    gy, gx = np.gradient(a, axis=(1, 2))
    magnitude = np.hypot(gx, gy)
    n, h, w, c = a.shape
    cells = magnitude.reshape(n, 4, h//4, 4, w//4, c)
    return np.concatenate([cells.mean((2,4)).reshape(n,-1), cells.std((2,4)).reshape(n,-1)],axis=1)


def fit_features(train, seed):
    raw = np.asarray(train, dtype=np.float64).reshape(len(train), -1) / 255
    pca = PCA(n_components=min(32, len(train)-1, raw.shape[1]), svd_solver='randomized', random_state=seed).fit(raw)
    texture_scaler = StandardScaler().fit(texture_features(train))
    pca_scaler = StandardScaler().fit(pca.transform(raw))
    return pca, texture_scaler, pca_scaler


def transform(images, pca, texture_scaler, pca_scaler):
    raw = np.asarray(images,dtype=np.float64).reshape(len(images),-1)/255
    return {'texture':texture_scaler.transform(texture_features(images)),
            'pixel_pca':pca_scaler.transform(pca.transform(raw))}


def energy(x, y):
    # Biased empirical energy distance; nonnegative, finite-sample baseline is nonzero.
    return float(2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean())


def compare(train, held, generated, seed, options):
    rng=np.random.default_rng(seed)
    n=min(len(held),len(generated),options['metric_rows'])
    if n<3:
        return {'status':'insufficient_samples'}
    si=rng.choice(len(generated),n,replace=False);hi=rng.choice(len(held),n,replace=False)
    x,y=generated[si],held[hi]
    xy,xx,yy=cdist(x,y),cdist(x,x),cdist(y,y)
    score=float(2*xy.mean()-xx.mean()-yy.mean())
    refs=[energy(train[rng.choice(len(train),n,replace=True)],y) for _ in range(options['reference_repeats'])]
    draws=[]
    for _ in range(min(options['bootstrap'],200)):
        a,b=rng.integers(n,size=n),rng.integers(n,size=n)
        draws.append(float(2*xy[np.ix_(a,b)].mean()-xx[np.ix_(a,a)].mean()-yy[np.ix_(b,b)].mean()))
    return {'status':'computed','metric':'biased_empirical_energy_distance','sample_count':n,
            'synthetic_discrepancy':score,'real_reference_median':float(np.median(refs)),
            'real_reference_range':[float(min(refs)),float(max(refs))],
            'synthetic_bootstrap_95_ci':np.quantile(draws,[.025,.975]).tolist(),
            'generated_local_indices':si.tolist(),'heldout_local_indices':hi.tolist(),
            'interpretation':'Smaller is closer in this feature space. No acceptance cutoff. Reference range is not a confidence interval; bootstrap conditions on one fitted generator.'}


def assess(train, held, samples, fit, options):
    args=fit['image_pca'],fit['texture_scaler'],fit['pixel_pca_scaler']
    t,h,s=[transform(a,*args) for a in (train,held,samples)]
    return {'status':'computed','version':'1.0',
            'features':'Local gradient magnitude mean/std on a 4x4 grid, and up to 32 PCA components of full RGB pixels. Both scalers and PCA fit real training only.',
            'pca_explained_variance_ratio':fit['image_pca'].explained_variance_ratio_.tolist(),
            'comparisons':{k:compare(t[k],h[k],s[k],options['seed']+37,options) for k in t},
            'limitations':['Not FID, canonical HOG, or a validated histology representation.',
                           'PCA may emphasize color/variance rather than clinically meaningful structure.',
                           'Energy distance assesses joint distributions only in the chosen feature representation.']}


def visual_packet(out, held, samples, held_labels, sample_labels, hi, gi, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(seed)
    conditional=held_labels is not None and sample_labels is not None
    groups=list(np.unique(np.concatenate([held_labels,sample_labels]))) if conditional else [None]
    entries=[];figures=[]
    for start in range(0,len(groups),3):
        batch=groups[start:start+3]
        fig,axes=plt.subplots(len(batch)*2,4,figsize=(10,4*len(batch)),squeeze=False)
        for j,label in enumerate(batch):
            for side,(images,labels,original) in enumerate(((held,held_labels,hi),(samples,sample_labels,gi))):
                pool=np.flatnonzero(labels==label) if conditional else np.arange(len(images))
                chosen=rng.choice(pool,min(4,len(pool)),replace=False)
                entries.append({'class':str(label) if conditional else 'unconditional','source':'real_holdout' if side==0 else 'generated',
                                'original_indices':np.asarray(original)[chosen].tolist()})
                for col,ax in enumerate(axes[2*j+side]):
                    ax.axis('off')
                    if col<len(chosen):
                        a=images[chosen[col]]
                        ax.imshow(a.squeeze(),cmap='gray' if a.ndim==2 or a.shape[-1]==1 else None,vmin=0,vmax=255,interpolation='nearest')
                        ax.set_title(f"{'Real' if side==0 else 'Generated'} {original[chosen[col]]}",fontsize=8)
                    else: ax.text(.1,.5,'No sample',fontsize=8)
                axes[2*j+side,0].set_ylabel(f'Class {label}',fontsize=8)
                axes[2*j+side,0].text(-.4,.5,f'Class {label}' if conditional else 'Unconditional',transform=axes[2*j+side,0].transAxes,rotation=90,va='center',fontsize=8)
        fig.suptitle('Fixed-seed held-out real and generated samples; no cherry-picking',fontsize=10)
        fig.tight_layout(rect=(0.04,0,1,0.97),h_pad=2);name=f'visual_samples_{start//3+1}.png';fig.savefig(out/name,dpi=140);plt.close(fig);figures.append(name)
    packet={'status':'prepared_for_review','selection':'Up to four samples per source and class, seeded uniformly without replacement; no labels inferred for unconditional outputs.',
            'figures':figures,'samples':entries,'review_status':'not_yet_reviewed',
            'checklist':['Compare color and stain appearance.','Inspect fine texture, blur and repeated patterns.','Compare diversity within each class.','Do not infer tissue-label correctness or clinical validity without appropriate expertise.']}
    save(out/'visual_review.json',packet)
    lines=['# Visual review packet','','Status: not yet reviewed. Figure generation alone is not visual inspection.','']
    lines += [f'![Samples]({f})' for f in figures]+['']+[f'- {s}' for s in packet['checklist']]
    (out/'visual_review.md').write_text('\n'.join(lines)+'\n')
    return packet
