"""Exact, column-wise profiling without densifying sparse matrices."""

from collections import Counter
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy import sparse

from .data import Dataset, InputError


def _numeric_stats(values, implicit_zeros=0):
    values = np.asarray(values, dtype=np.float64)
    missing = int(np.isnan(values).sum())
    infinite = int(np.isinf(values).sum())
    finite = values[np.isfinite(values)]
    zeros = int((finite == 0).sum()) + implicit_zeros
    count = len(finite) + implicit_zeros
    low = float(finite.min()) if len(finite) else None
    high = float(finite.max()) if len(finite) else None
    if implicit_zeros:
        low = min(0.0, low) if low is not None else 0.0
        high = max(0.0, high) if high is not None else 0.0
    return {
        "missing": missing, "infinite": infinite, "zeros": zeros,
        "finite_min": low, "finite_max": high,
        "constant_among_finite": bool(count and low == high),
        "all_missing": missing == len(values) + implicit_zeros,
        "nonnegative_finite": bool(count and low >= 0),
        "integer_valued_finite": bool(count and np.equal(finite, np.floor(finite)).all()),
    }


def profile_dataset(dataset: Dataset):
    X = dataset.X
    n, p = X.shape
    is_sparse = sparse.issparse(X)
    matrix = X.tocsc() if is_sparse else X
    if is_sparse and X.dtype.kind not in "biuf":
        raise InputError("Sparse measurements must be real numeric values.")
    features = []
    for j in range(p):
        entry = {"position": j, "name": str(dataset.var.index[j])}
        if is_sparse:
            values = matrix.data[matrix.indptr[j]:matrix.indptr[j + 1]]
            kind, dtype = "numeric", str(matrix.dtype)
            stats = _numeric_stats(values, n - len(values))
        else:
            series = X.iloc[:, j] if isinstance(X, pd.DataFrame) else pd.Series(X[:, j])
            dtype = str(series.dtype)
            if pd.api.types.is_complex_dtype(series.dtype):
                raise InputError("Complex-valued features are not supported.")
            if pd.api.types.is_numeric_dtype(series.dtype):
                kind = "boolean" if pd.api.types.is_bool_dtype(series.dtype) else "numeric"
                stats = _numeric_stats(series.to_numpy(dtype=float, na_value=np.nan))
            else:
                kind = "datetime" if pd.api.types.is_datetime64_any_dtype(series.dtype) else "categorical_or_text"
                stats = {"missing": int(series.isna().sum()), "infinite": 0,
                         "unique_nonmissing": int(series.nunique(dropna=True)),
                         "constant_among_observed": series.nunique(dropna=True) == 1,
                         "all_missing": bool(series.isna().all())}
        features.append({**entry, "kind": kind, "dtype": dtype, **stats})

    numeric = [f for f in features if f["kind"] in ("numeric", "boolean")]
    missing = sum(f["missing"] for f in features)
    infinite = sum(f["infinite"] for f in features)
    numeric_cells = n * len(numeric)
    observed_numeric = numeric_cells - sum(f["missing"] for f in numeric)
    zero_count = sum(f["zeros"] for f in numeric)
    warnings = list(dataset.ambiguities)
    if missing:
        warnings.append("Missing measurements require an explicit preprocessing decision.")
    if infinite:
        warnings.append("Infinite measurements must be resolved before reduction.")
    if len(numeric) != p:
        warnings.append("Non-numeric features require encoding or explicit exclusion.")
    if n < 3:
        warnings.append("Fewer than three observations: a 2D exploratory embedding may not be meaningful.")
    label_info = {"available": dataset.labels is not None, "used_as_features": False, "columns": []}
    if dataset.labels is not None:
        label_info["columns"] = [{"name": str(c), "missing": int(dataset.labels[c].isna().sum()),
                                  "unique_nonmissing": int(dataset.labels[c].nunique())} for c in dataset.labels]
    return {
        "schema_version": "1.0", "stage": "input_profile",
        "n_observations": n, "n_features": p,
        "orientation": "observations_by_features",
        "storage": "sparse_csr" if is_sparse else "dense",
        "feature_types": dict(Counter(f["kind"] for f in features)),
        "missing_values": missing, "missing_fraction": missing / (n * p),
        "infinite_values": infinite,
        "numeric_zero_fraction": zero_count / observed_numeric if observed_numeric else None,
        "zero_fraction_definition": "Numeric zeros / nonmissing numeric cells; includes implicit sparse zeros; infinities are nonmissing.",
        "constant_feature_positions": [f["position"] for f in features if f.get("constant_among_finite", f.get("constant_among_observed", False))],
        "all_missing_feature_positions": [f["position"] for f in features if f["all_missing"]],
        "resource_estimates": {"dense_float64_matrix_bytes": n * p * 8, "one_dense_pairwise_float64_matrix_bytes": n * n * 8,
                               "note": "Array sizes only, not estimates of total algorithm memory."},
        "labels": label_info,
        "observation_metadata_columns": [str(c) for c in dataset.obs.columns],
        "feature_metadata_columns": [str(c) for c in dataset.var.columns],
        "assumptions": dataset.assumptions, "ambiguities": dataset.ambiguities,
        "warnings": warnings,
        "requires_clarification": bool(dataset.ambiguities),
        "provenance": dataset.provenance,
        "software": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "scipy", "anndata")}},
        "features": features,
    }


def render_summary(profile):
    description = ("This profiles preprocessed measurements; no dimension reduction has run."
                   if "preprocessing_plan" in profile["provenance"] else
                   "This is an inspection report; no reduction or statistical preprocessing has run.")
    lines = ["# Dataset profile", "", description, "",
             f"- Observations: {profile['n_observations']:,}", f"- Features: {profile['n_features']:,}",
             f"- Storage: {profile['storage']}", f"- Missing values: {profile['missing_values']:,}",
             f"- Infinite values: {profile['infinite_values']:,}",
             f"- Numeric zero fraction: {profile['numeric_zero_fraction']}",
             f"- Labels available: {profile['labels']['available']}",
             f"- Requires clarification: {profile['requires_clarification']}", ""]
    for title, key in [("Assumptions", "assumptions"), ("Warnings and unresolved questions", "warnings")]:
        lines.extend([f"## {title}", ""] + [f"- {v}" for v in profile[key]] + [""])
    lines.extend(["## Reproducibility", "", "See profile.json for input SHA-256 hashes, software versions, per-feature statistics, and memory estimates.", ""])
    return "\n".join(lines)
