"""Generative-model inspection; no fitting or preprocessing is performed."""

import platform
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version


from .base_profile import profile_dataset as base_profile
from .data import InputError


def compute_profile():
    total = available = None
    try:
        import psutil
        memory = psutil.virtual_memory()
        total, available = memory.total, memory.available
    except ImportError:
        if platform.system() == "Darwin":
            try:
                total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL))
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
        elif hasattr(os, "sysconf"):
            try:
                total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            except (ValueError, OSError):
                pass
    result = {
        "system": platform.system(), "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": total,
        "available_memory_bytes_at_inspection": available,
        "memory_note": "Available system memory is a snapshot, not a training allocation or GPU memory budget.",
        "torch_version": None, "mps_available": None, "cuda_available": None,
        "suggested_device": "cpu", "backend_probe_error": None,
    }
    try:
        import torch
        result.update(torch_version=torch.__version__,
                      mps_available=bool(torch.backends.mps.is_available()),
                      cuda_available=bool(torch.cuda.is_available()))
        result["suggested_device"] = "cuda" if result["cuda_available"] else "mps" if result["mps_available"] else "cpu"
    except ImportError:
        result["backend_probe_error"] = "PyTorch is not installed; accelerator availability is unverified."
    except Exception as exc:
        result["backend_probe_error"] = f"PyTorch probe failed: {type(exc).__name__}: {exc}"
    return result


def profile_dataset(dataset, objective):
    if not objective.strip():
        raise InputError("A nonempty user objective is required.")
    result = base_profile(dataset)
    result.update(schema_version="2.0", stage="generative_input_inspection",
                  objective={"text": objective.strip(), "interpretation_status": "pending_planner"},
                  compute=compute_profile())
    fmt = dataset.provenance["format"]
    result["modality"] = "image" if fmt == "medmnist_npz" else "single_cell_matrix" if fmt in ("10x_mtx", "h5ad") else "tabular"
    result["modality_evidence"] = "Input format; H5AD biological semantics still require confirmation."
    result["image_shape"] = dataset.provenance.get("image_shape")
    result["resource_estimates"] = {
        "dense_float32_matrix_bytes": result["n_observations"] * result["n_features"] * 4,
        "dense_float64_matrix_bytes": result["n_observations"] * result["n_features"] * 8,
        "note": "Matrix storage only; excludes parameters, gradients, optimizer state and activations.",
    }
    numeric = [f for f in result["features"] if f["kind"] in ("numeric", "boolean")]
    count_compatible = (len(numeric) == result["n_features"] and not result["missing_values"]
                        and not result["infinite_values"]
                        and all(f["nonnegative_finite"] and f["integer_valued_finite"] for f in numeric))
    result["value_semantics"] = {
        "nonnegative_integer_matrix": bool(count_compatible),
        "raw_count_evidence": "10x input convention and nonnegative integer values" if fmt == "10x_mtx" and count_compatible else "not established",
        "note": "Integer values alone do not establish count semantics; image pixels are not gene counts.",
    }
    if result["modality"] == "image":
        result["value_semantics"]["image_dtype"] = str(dataset.X.dtype)
        result["value_semantics"]["pixel_range"] = (
            [float(dataset.X.min()), float(dataset.X.max())]
            if not result["missing_values"] and not result["infinite_values"] else None
        )
    for info in result["labels"]["columns"]:
        values = dataset.labels[info["name"]].dropna()
        counts = values.value_counts()
        info["class_counts"] = [{"value": str(k), "count": int(v)} for k, v in counts.items()]
        info["majority_to_minority_ratio"] = float(counts.max() / counts.min()) if len(counts) else None
        info["balance_note"] = "Observed label frequencies; class interpretation belongs to the planner."
    result["split_policy"] = {
        "inspected_split": dataset.provenance.get("split"),
        "splits_combined": False,
        "training_split_created": False,
        "requirement": "Fit preprocessing and models on training data only; use validation for selection and reserve test data for final evaluation.",
    }
    result["warnings"] = [w.replace("before reduction", "before modeling").replace("a 2D exploratory embedding may not be meaningful", "generative modeling cannot be assessed reliably") for w in result["warnings"]]
    if not result["labels"]["available"]:
        result["warnings"].append("No explicit labels available; supervised class-balancing evaluation requires justified labels.")
    if fmt == "medmnist_npz" and dataset.provenance["split"] != "train":
        result["warnings"].append("This is not the training split; do not use it for training or fitted preprocessing.")
    for package in ("psutil", "genmodel-agent"):
        try:
            result["software"][package] = version(package)
        except PackageNotFoundError:
            result["software"][package] = "not installed (source checkout)"
    return result


def render_summary(p):
    zero = p["numeric_zero_fraction"]
    lines = ["# Generative modeling dataset inspection", "",
             "Inspection only: no model selection, training, generation, or fitted preprocessing has run.", "",
             "## User objective", "", p["objective"]["text"], "",
             "Objective interpretation is pending the planning milestone.", "",
             "## Dataset", "",
             f"- Modality: {p['modality']}",
             f"- Observations: {p['n_observations']:,}", f"- Features: {p['n_features']:,}",
             f"- Feature types: {p['feature_types']}", f"- Storage: {p['storage']}",
             f"- Image shape: {p['image_shape']}",
             f"- Inspected split: {p['split_policy']['inspected_split'] or 'unsplit input'}",
             f"- Missing values: {p['missing_values']:,}", f"- Infinite values: {p['infinite_values']:,}",
             f"- Numeric zero fraction: {zero}",
             f"- Constant features: {len(p['constant_feature_positions']):,}",
             f"- Count evidence: {p['value_semantics']['raw_count_evidence']}",
             f"- Labels available: {p['labels']['available']}", ""]
    for label in p["labels"]["columns"]:
        lines += [f"### Label: {label['name']}", "",
                  f"Missing: {label['missing']}; distinct observed values: {label['unique_nonmissing']}; majority/minority ratio: {label['majority_to_minority_ratio']}", ""]
        lines += [f"- {c['value']}: {c['count']:,}" for c in label["class_counts"][:30]]
        if len(label["class_counts"]) > 30:
            lines.append("- Remaining label counts are in profile.json.")
        lines.append("")
    c = p["compute"]
    def gib(value):
        return "unavailable" if value is None else f"{value / 2**30:.1f} GiB"
    lines += ["## Compute", "", f"- System: {c['system']} ({c['architecture']})",
              f"- Physical memory: {gib(c['physical_memory_bytes'])}",
              f"- Available memory snapshot: {gib(c['available_memory_bytes_at_inspection'])}",
              f"- PyTorch: {c['torch_version'] or 'unavailable'}",
              f"- Suggested device: {c['suggested_device']}",
              f"- Backend probe: {c['backend_probe_error'] or 'successful; individual model operations remain untested'}", "",
              "## Split safeguards", "", p["split_policy"]["requirement"], ""]
    for title, key in (("Assumptions", "assumptions"), ("Warnings", "warnings")):
        lines += [f"## {title}", ""] + [f"- {w}" for w in p[key]] + [""]
    lines += ["## Reproducibility", "", "profile.json contains source SHA-256 hashes, package versions, per-feature statistics, label counts, and matrix memory estimates. Source paths and label values are included. No external AI service was called.", ""]
    return "\n".join(lines)
