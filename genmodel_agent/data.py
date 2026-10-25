"""Format-based loading. No inference of scientific meaning from filenames."""

from dataclasses import dataclass, field
from pathlib import Path
import gzip
import hashlib

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


class InputError(ValueError):
    """An unsupported or ambiguous input needs user clarification."""


@dataclass
class Dataset:
    X: object
    obs: pd.DataFrame
    var: pd.DataFrame
    labels: pd.DataFrame | None = None
    provenance: dict = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)

    @property
    def images(self):
        """Recover the original image tensor without discarding spatial layout."""
        shape = self.provenance.get("image_shape")
        return None if shape is None else self.X.reshape((len(self.obs), *shape))

    def __post_init__(self):
        if len(self.X.shape) != 2 or min(self.X.shape) == 0:
            raise InputError("Measurements must be a nonempty observations-by-features matrix.")
        if self.X.shape != (len(self.obs), len(self.var)):
            raise InputError("Measurement dimensions do not match observation/feature metadata.")
        if not self.obs.index.is_unique:
            raise InputError("Observation identifiers must be unique.")
        if self.labels is not None and not self.labels.index.equals(self.obs.index):
            raise InputError("Labels must align exactly with observation identifiers.")


def _metadata(n, prefix):
    return pd.DataFrame(index=pd.Index([f"{prefix}_{i}" for i in range(n)], name=prefix))


def _fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _tenx_file(folder, names):
    matches = [folder / name for name in names if (folder / name).is_file()]
    if len(matches) != 1:
        raise InputError(f"Expected exactly one of {names} in {folder}.")
    return matches[0]


def load_dataset(path, *, label_column=None, id_column=None, split=None, orientation=None):
    """Read CSV/TSV, MedMNIST NPZ, AnnData H5AD or a 10x MTX directory.

    CSV assumes a header row; orientation must be explicit. No transformations,
    imputation, normalization, sampling or label-driven feature selection occur.
    """
    path = Path(path)
    if not path.exists():
        raise InputError(f"Input does not exist: {path}")
    if orientation not in (None, "rows", "columns"):
        raise InputError("orientation must be rows or columns.")
    sources = []
    assumptions, ambiguities = [], []
    labels = None
    detail = {}

    if path.is_dir():
        if any(v is not None for v in (label_column, id_column, split, orientation)):
            raise InputError("10x layout is explicit; column, split and orientation overrides do not apply.")
        matrix = _tenx_file(path, ["matrix.mtx", "matrix.mtx.gz"])
        genes = _tenx_file(path, ["genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz"])
        barcodes = _tenx_file(path, ["barcodes.tsv", "barcodes.tsv.gz"])
        opener = gzip.open if matrix.suffix == ".gz" else open
        with opener(matrix, "rb") as handle:
            X = sparse.csr_matrix(mmread(handle).T)
        var = pd.read_csv(genes, sep="\t", header=None, dtype=str, keep_default_na=False)
        obs = pd.read_csv(barcodes, sep="\t", header=None, dtype=str, keep_default_na=False)
        if var.shape[1] not in (2, 3) or obs.shape[1] != 1:
            raise InputError("Expected 10x genes/features (2 or 3 columns) and one barcode column.")
        var.columns = ["feature_id", "feature_name", "feature_type"][:var.shape[1]]
        var.index = pd.Index(var.feature_id, name="feature")
        obs = pd.DataFrame(index=pd.Index(obs[0], name="observation"))
        sources = [matrix, genes, barcodes]
        detail = {"format": "10x_mtx", "orientation_evidence": "10x features-by-barcodes convention"}
        assumptions.append("Transposed 10x matrix to observations by features; counts remain unchanged.")
        if "feature_type" in var and var.feature_type.nunique() > 1:
            ambiguities.append("Multiple 10x feature types: choose a modality before analysis.")

    elif path.suffix.lower() in (".csv", ".tsv"):
        if split is not None:
            raise InputError("split applies only to MedMNIST NPZ.")
        if orientation is None:
            raise InputError("CSV/TSV orientation is ambiguous. Specify --orientation rows or columns.")
        table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
        if label_column == id_column and label_column is not None:
            raise InputError("Label and ID columns must differ.")
        if orientation == "columns" and label_column is not None:
            raise InputError("Column-oriented tables do not support --label-column; transpose/export labels separately.")
        for name in (label_column, id_column):
            if name is not None and name not in table:
                raise InputError(f"Column not found: {name}")
        if id_column is not None:
            if table[id_column].isna().any() or table[id_column].duplicated().any():
                raise InputError("ID column must be complete and unique.")
            table = table.set_index(id_column)
        if orientation == "columns":
            table = table.T
        obs = pd.DataFrame(index=table.index.copy())
        if label_column is not None:
            labels = table[[label_column]].copy()
            table = table.drop(columns=label_column)
        X = table
        var = pd.DataFrame(index=pd.Index(table.columns, name="feature"))
        candidates = [str(c) for c in table if str(c).lower() in {"label", "target", "class", "group", "id", "sample_id", "cell_id"}]
        if candidates:
            ambiguities.append(f"Possible label/identifier columns retained pending clarification: {candidates}")
        assumptions.append("First line interpreted as column headers; standard pandas missing-value tokens apply.")
        detail = {"format": "delimited", "orientation_evidence": f"user specified observations in {orientation}"}

    elif path.suffix.lower() == ".npz":
        if any(v is not None for v in (label_column, id_column, orientation)):
            raise InputError("MedMNIST layout and labels are explicit; column/orientation overrides do not apply.")
        with np.load(path, allow_pickle=False) as archive:
            available = [s for s in ("train", "val", "test") if f"{s}_images" in archive]
            if not available:
                raise InputError("NPZ must follow MedMNIST <split>_images / <split>_labels layout.")
            if split is None and len(available) > 1:
                raise InputError(f"Choose --split from {available}; splits are never silently combined.")
            split = split or available[0]
            if split not in available or f"{split}_labels" not in archive:
                raise InputError("Selected split is absent or has no explicit labels.")
            images, targets = archive[f"{split}_images"], archive[f"{split}_labels"]
            if images.ndim not in (3, 4, 5) or images.dtype.kind not in "biuf":
                raise InputError("Expected numeric MedMNIST images with an observation axis followed by image axes.")
            if targets.ndim == 1:
                targets = targets[:, None]
            if targets.ndim != 2 or len(targets) != len(images):
                raise InputError("Labels do not match image observations.")
            X = images.reshape(len(images), -1)
            obs, var = _metadata(len(X), split), _metadata(X.shape[1], "pixel")
            labels = pd.DataFrame(targets, index=obs.index, columns=[f"label_{i}" for i in range(targets.shape[1])])
            detail = {"format": "medmnist_npz", "split": split, "image_shape": list(images.shape[1:]), "orientation_evidence": "MedMNIST array convention"}
            assumptions.append("Flattened image axes in C order; pixel values are not scaled and spatial layout is recorded.")

    elif path.suffix.lower() == ".h5ad":
        if any(v is not None for v in (id_column, split, orientation)):
            raise InputError("H5AD layout is explicit; ID, split and orientation overrides do not apply.")
        import anndata
        adata = anndata.read_h5ad(path)
        if adata.X is None:
            raise InputError("H5AD has no X matrix.")
        X, obs, var = adata.X, adata.obs.copy(), adata.var.copy()
        if label_column is not None:
            if label_column not in obs:
                raise InputError(f"Observation label column not found: {label_column}")
            labels = obs[[label_column]].copy()
        detail = {"format": "h5ad", "orientation_evidence": "AnnData observations-by-variables convention", "matrix_source": "X", "available_layers": list(adata.layers.keys()), "raw_available": adata.raw is not None}
        ambiguities.append("H5AD X preprocessing history is unknown; verify whether values are raw counts or already transformed.")
    else:
        raise InputError("Supported inputs: CSV, TSV, MedMNIST NPZ, H5AD, or extracted 10x MTX directory.")

    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        X.sum_duplicates()
        X.eliminate_zeros()
    return Dataset(X, obs, var, labels, {**detail, "sources": [_fingerprint(p) for p in (sources or [path])]}, assumptions, ambiguities)
