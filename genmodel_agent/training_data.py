"""Source verification, leakage-safe splits and training-only preprocessing."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

from .data import load_dataset
from .planning import require, save


@dataclass
class TrainingData:
    train: object
    val: object
    train_labels: np.ndarray | None
    val_labels: np.ndarray | None
    shape: tuple
    classes: list
    metadata: dict

    def batch(self, indices, split="train"):
        matrix = self.train if split == "train" else self.val
        x = matrix[indices]
        if sparse.issparse(x):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float32)
        if self.metadata["kind"] == "image":
            if x.ndim == 3:
                x = x[..., None]
            x = x.transpose(0, 3, 1, 2) / 255.0
            if self.metadata.get("scaling") == "uint8 / 127.5 - 1":
                x = 2*x - 1
        labels = self.train_labels if split == "train" else self.val_labels
        return np.ascontiguousarray(x), None if labels is None else labels[indices]


def gene_selection(train, count):
    """Rank by standardized log dispersion within equal-frequency mean bins."""
    x = train.astype(np.float64)
    mean = np.asarray(x.mean(axis=0)).ravel()
    second = np.asarray(x.power(2).mean(axis=0)).ravel() if sparse.issparse(x) else (x*x).mean(0)
    variance = np.maximum(second - mean * mean, 0)
    eligible = np.flatnonzero((mean > 0) & (variance > 1e-12))
    require(len(eligible) >= count, f"Only {len(eligible)} nonconstant training genes for requested {count}; replan rather than silently changing feature count")
    ordered = eligible[np.argsort(mean[eligible], kind="stable")]
    scores = np.full(len(mean), -np.inf)
    dispersion = np.log1p(variance / np.maximum(mean, 1e-12))
    for group in np.array_split(ordered, min(20, max(1, len(ordered)//20))):
        sd = dispersion[group].std()
        scores[group] = (dispersion[group] - dispersion[group].mean()) / max(sd, 1e-8)
    ranking = eligible[np.argsort(-scores[eligible], kind="stable")]
    return np.sort(ranking[:count]), scores, mean


def subset(indices, labels, limit, seed):
    if limit is None or len(indices) <= limit:
        return indices
    if labels is not None:
        result, _ = train_test_split(indices, train_size=limit, random_state=seed, stratify=labels[indices])
        return np.sort(result)
    return np.sort(np.random.default_rng(seed).choice(indices, limit, replace=False))


def encode_labels(train, val, conditional):
    if not conditional:
        return None, None, []
    require(train is not None and val is not None, "Conditional trainer requires train and validation labels")
    require(not np.any(np.equal(train, None)) and not np.any(np.equal(val, None)), "Missing labels")
    classes = np.unique(train)
    require(2 <= len(classes) <= 100, "Conditional training requires 2..100 classes")
    require(np.isin(val, classes).all(), "Validation contains a class absent from training")
    return np.searchsorted(classes, train).astype(np.int64), np.searchsorted(classes, val).astype(np.int64), classes.tolist()


def verify_source(data, profile):
    expected = sorted(s["sha256"] for s in profile["provenance"]["sources"])
    actual = sorted(s["sha256"] for s in data.provenance["sources"])
    require(expected == actual, "Dataset hashes differ from the profile used for planning")
    require(list(data.X.shape) == [profile["n_observations"], profile["n_features"]], "Dataset dimensions differ from planned profile")


def prepare(dataset_path, profile, plan, config, out, pilot=False, max_bytes=6*2**30):
    seed = plan["split"]["seed"]
    fmt = profile["provenance"]["format"]
    conditional = config["conditional"]
    meta = {"fit_on": "train", "pilot": pilot, "test_used_for_training_or_selection": False}
    if config["name"] in ("image_vae", "image_gan"):
        require(fmt == "medmnist_npz", "Image execution requires MedMNIST NPZ")
        with np.load(dataset_path, allow_pickle=False) as archive:
            require(all(k in archive.files for k in ("train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels")), "Official train, validation and test keys are required")
            # Deliberately do not access test arrays.
        data = load_dataset(dataset_path, split="train")
        verify_source(data, profile)
        val_data = load_dataset(dataset_path, split="val")
        train, val = data.images, val_data.images
        shape = tuple(train.shape[1:])
        require(tuple(val.shape[1:]) == shape, "Validation image shape mismatch")
        require(len(shape) in (2, 3) and (len(shape) == 2 or shape[-1] in (1, 3)), "Unsupported image channels")
        require(all(4 <= d <= 64 and d % 4 == 0 for d in shape[:2]), "Trainer supports dimensions divisible by 4, up to 64")
        require(train.dtype == np.uint8 and val.dtype == np.uint8, "Image scaling contract requires uint8 pixels; replan for other encodings")
        require(train.nbytes + val.nbytes < max_bytes // 2, "Image arrays exceed preprocessing memory allowance")
        require(len(data.labels.columns) == 1 and len(val_data.labels.columns) == 1, "Image executor currently supports single categorical labels")
        raw_y, raw_v = data.labels.iloc[:, 0].to_numpy(), val_data.labels.iloc[:, 0].to_numpy()
        train_i = subset(np.arange(len(train)), raw_y, 512 if pilot else None, seed)
        val_i = subset(np.arange(len(val)), raw_v, 128 if pilot else None, seed + 1)
        y, v, classes = encode_labels(raw_y[train_i], raw_v[val_i], conditional)
        train, val = train[train_i], val[val_i]
        meta.update(kind="image", scaling="uint8 / 127.5 - 1" if config["name"] == "image_gan" else "uint8 / 255", original_shape=list(shape),
                    source_hashes=[s["sha256"] for s in data.provenance["sources"]],
                    official_test="Reserved; arrays not loaded", label_column="label_0")
        np.savez_compressed(out / "splits.npz", train_indices=train_i, validation_indices=val_i)
    else:
        require(fmt == "10x_mtx", "Count execution currently requires verified raw 10x MTX; H5AD needs a raw-count contract")
        data = load_dataset(dataset_path)
        verify_source(data, profile)
        x = data.X
        values = x.data if sparse.issparse(x) else np.asarray(x)
        require(np.isfinite(values).all() and (values >= 0).all() and np.equal(values, np.floor(values)).all(), "Counts must be finite nonnegative integers")
        require((np.asarray(x.sum(1)).ravel() > 0).all(), "Zero-total cells require an explicit filtering decision")
        raw_y = None if data.labels is None else data.labels.iloc[:, 0].to_numpy()
        stratify = raw_y if plan["split"]["strategy"] == "stratified_holdout" else None
        indices = np.arange(x.shape[0])
        remain, test_i = train_test_split(indices, test_size=plan["split"]["test_fraction"], random_state=seed, stratify=stratify)
        fraction = plan["split"]["validation_fraction"] / (1 - plan["split"]["test_fraction"])
        train_i, val_i = train_test_split(remain, test_size=fraction, random_state=seed + 1,
                                        stratify=None if stratify is None else raw_y[remain])
        full_train_i, full_val_i = train_i.copy(), val_i.copy()
        train_i = subset(train_i, raw_y, 512 if pilot else None, seed + 2)
        val_i = subset(val_i, raw_y, 128 if pilot else None, seed + 3)
        require(config["selected_features"] * (len(train_i)+len(val_i)) * 4 < max_bytes // 2, "Selected count matrix exceeds memory allowance")
        features, scores, means = gene_selection(x[train_i], config["selected_features"])
        train, val = x[train_i][:, features], x[val_i][:, features]
        if config["name"] == "count_vae":
            require((np.asarray(train.sum(1)).ravel() > 0).all() and (np.asarray(val.sum(1)).ravel() > 0).all(), "Selected gene panel creates zero-total cells; replan with more features")
        y, v, classes = encode_labels(None if raw_y is None else raw_y[train_i], None if raw_y is None else raw_y[val_i], conditional)
        shape = (len(features),)
        meta.update(kind="counts", selected_feature_positions=features.tolist(),
                    selected_feature_ids=[str(data.var.index[i]) for i in features],
                    gene_selection="Training-only standardized log1p variance/mean dispersion in mean bins; constant genes excluded",
                    selected_panel_library_size="Counts summed over selected genes, not whole transcriptome",
                    source_hashes=[s["sha256"] for s in data.provenance["sources"]])
        np.savez_compressed(out / "gene_selection.npz", positions=features, scores=scores[features], training_means=means[features])
        np.savez_compressed(out / "splits.npz", train_indices=train_i, validation_indices=val_i,
                            test_indices=test_i, full_train_indices=full_train_i, full_validation_indices=full_val_i)
    meta.update(train_observations=len(train_i), validation_observations=len(val_i),
                classes=classes, shape=list(shape))
    save(out / "preprocessing.json", meta)
    return TrainingData(train, val, y, v, shape, classes, meta)
