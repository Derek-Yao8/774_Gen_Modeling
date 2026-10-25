"""VAE training, checkpoint selection, and generated artifacts (no utility claims)."""
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch

from .capabilities import COMMON, TRAINERS
from .models import build_model
from .planning import require, save
from .training_data import prepare


class ResourceLimit(RuntimeError):
    pass


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_limits(deadline, max_bytes, device="cpu"):
    if time.monotonic() >= deadline:
        raise ResourceLimit("wall_time_limit")
    if psutil.Process().memory_info().rss > max_bytes:
        raise ResourceLimit("rss_memory_limit")
    if device == "mps" and torch.mps.driver_allocated_memory() > max_bytes:
        raise ResourceLimit("mps_memory_limit")


def tensors(data, indices, device, split="train"):
    x, y = data.batch(indices, split)
    return torch.from_numpy(x).to(device), None if y is None else torch.from_numpy(y).to(device)


def choose_device(model, data, requested):
    candidates = [requested] if requested != "auto" else (
        ["cuda", "cpu"] if torch.cuda.is_available() else ["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"])
    attempts = []
    for device in candidates:
        try:
            if device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS unavailable in this process")
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable in this process")
            probe = copy.deepcopy(model).to(device)
            x, y = tensors(data, np.arange(min(8, data.train.shape[0])), device)
            loss, _, _ = probe.loss(x, y)
            require(torch.isfinite(loss).item(), "Nonfinite device probe loss")
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in probe.parameters()), "Nonfinite device probe gradient")
            attempts.append({"device": device, "status": "forward_backward_passed"})
            del probe, x, y, loss
            return device, attempts
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            attempts.append({"device": device, "status": "failed", "error": str(exc)})
            if requested != "auto":
                raise RuntimeError(f"Requested device probe failed: {attempts}") from exc
    raise RuntimeError(f"No usable device: {attempts}")


def validate_epoch(model, data, batch_size, device, deadline, max_bytes, seed):
    model.eval()
    totals = np.zeros(3)
    n = data.val.shape[0]
    noise_rng = np.random.default_rng(seed + 123)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            check_limits(deadline, max_bytes, device)
            indices = np.arange(start, min(start+batch_size, n))
            x, y = tensors(data, indices, device, "val")
            noise = torch.from_numpy(noise_rng.standard_normal((len(indices), model.latent_dim)).astype(np.float32)).to(device)
            losses = model.loss(x, y, noise)
            values = np.array([v.item() for v in losses])
            require(np.isfinite(values).all(), "Nonfinite validation loss")
            totals += values * len(indices)
    return (totals / n).tolist()


def save_checkpoint(path, model, optimizer, config, data, epoch, seed, plan_sha):
    payload = {"format_version": 1, "trainer_version": TRAINERS[config["name"]]["version"],
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer_state_dict": optimizer if isinstance(optimizer, dict) else optimizer.state_dict(), "config": config,
               "shape": list(data.shape), "classes": data.classes, "epoch": epoch,
               "seed": seed, "plan_sha256": plan_sha, "preprocessing": data.metadata}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def generate(model, data, config, out, seed, device, deadline, max_bytes):
    n, batch_size = config["generate_samples"], config["batch_size"]
    rng = np.random.default_rng(seed + 2000)
    classes = len(data.classes)
    y_all = np.arange(n, dtype=np.int64) % classes if classes else None
    # Bound accumulated arrays (including mean outputs and concatenation copies).
    require(n * math.prod(data.shape) * 32 < max_bytes // 2, "Requested generated arrays exceed memory allowance")
    means, samples, libraries = [], [], []
    train_library = np.asarray(data.train.sum(1)).ravel() if config["name"] == "count_vae" else None
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            check_limits(deadline, max_bytes, device)
            size = min(batch_size, n-start)
            z = torch.from_numpy(rng.standard_normal((size, model.latent_dim)).astype(np.float32)).to(device)
            y = None if y_all is None else torch.from_numpy(y_all[start:start+size]).to(device)
            if config["name"] == "image_vae":
                mean = model.decode(z, y).cpu().numpy().transpose(0, 2, 3, 1)
                if len(data.shape) == 2:
                    mean = mean[..., 0]
                draw = mean + rng.normal(0, model.sigma, size=mean.shape)
                samples.append(np.rint(np.clip(draw, 0, 1) * 255).astype(np.uint8))
                means.append(mean)
            else:
                if y_all is None:
                    library = rng.choice(train_library, size)
                else:
                    library = np.array([rng.choice(train_library[data.train_labels == label]) for label in y_all[start:start+size]])
                mean = model.decode(z, torch.from_numpy(library.astype(np.float32)).to(device), y).cpu().numpy()
                theta = model.dispersion.cpu().numpy()
                # Gamma-Poisson mixture is exactly the NB2 distribution, sampled on CPU.
                rate = rng.gamma(shape=theta, scale=mean / theta)
                require(np.isfinite(rate).all() and (rate < 1e12).all(), "Invalid generated count rate")
                samples.append(rng.poisson(rate).astype(np.int64))
                means.append(mean)
                libraries.append(library)
    shape = (0, *data.shape)
    output = {"samples": np.concatenate(samples) if samples else np.empty(shape, dtype=np.uint8 if config["name"] == "image_vae" else np.int64),
              "decoder_means": np.concatenate(means) if means else np.empty(shape, dtype=np.float32)}
    if y_all is not None:
        output["label_indices"] = y_all
        output["labels"] = np.asarray(data.classes)[y_all]
    if config["name"] == "count_vae":
        output["selected_feature_ids"] = np.asarray(data.metadata["selected_feature_ids"], dtype=str)
        output["conditioning_library_sizes"] = np.concatenate(libraries) if libraries else np.empty(0)
    np.savez_compressed(out / "generated.npz", **output)
    return output


def representations(model, data, batch_size, out, device, deadline, max_bytes):
    output = {}
    model.eval()
    with torch.no_grad():
        for split, matrix in (("train", data.train), ("val", data.val)):
            vectors = []
            for start in range(0, matrix.shape[0], batch_size):
                check_limits(deadline, max_bytes, device)
                x, y = tensors(data, np.arange(start, min(start+batch_size, matrix.shape[0])), device, split)
                mu, _ = model.encode(x, y)
                vectors.append(mu.cpu().numpy())
            output[split] = np.concatenate(vectors)
    np.savez_compressed(out / "representations.npz", **output)


def plots(out, history, generated, image):
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h["epoch"] for h in history], [h["train_negative_elbo"] for h in history], label="Train")
    ax.plot([h["epoch"] for h in history], [h["validation_negative_elbo"] for h in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Negative ELBO per observation", title="Training diagnostic — not synthetic-data quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=140)
    plt.close(fig)
    if image and len(generated["samples"]):
        count = min(16, len(generated["samples"]))
        fig, axes = plt.subplots(2, count, figsize=(max(8, count), 3), squeeze=False)
        for row, key in enumerate(("decoder_means", "samples")):
            for i in range(count):
                axes[row, i].imshow(generated[key][i].squeeze(), cmap="gray", vmin=0, vmax=1 if row == 0 else 255)
                axes[row, i].axis("off")
        fig.suptitle("Decoder means (top); Gaussian draws clipped to pixels (bottom)")
        fig.tight_layout()
        fig.savefig(out / "generated_grid.png", dpi=140)
        plt.close(fig)


def train_job(job, out):
    from .flows import FLOW_NAMES
    if job["config"]["name"] in FLOW_NAMES:
        from .flow_trainer import train_job as train_flow
        return train_flow(job, out)
    if job["config"]["name"] == "image_gan":
        from .gan_trainer import train_job as train_gan
        return train_gan(job, out)
    started = time.monotonic()
    config, plan = job["config"], job["plan"]
    seed = plan["split"]["seed"]
    max_bytes = job["max_bytes"]
    deadline = started + job["wall_seconds"]
    # Reserve 20% of branch allowance for validation/checkpoint/generation artifacts.
    train_deadline = started + job["wall_seconds"] * 0.8
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    seed_all(seed)
    data = prepare(job["dataset"], job["profile"], plan, config, out, job["pilot"], max_bytes)
    check_limits(deadline, max_bytes)
    model = build_model(config, data.shape, len(data.classes))
    device, probes = choose_device(model, data, job["device"])
    save(out / "device.json", {"requested": job["device"], "selected": device, "probes": probes})
    seed_all(seed)  # Probe must not advance training randomness.
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history, best, best_epoch, stale = [], float("inf"), None, 0
    stop_reason = "epochs_completed"
    rng = np.random.default_rng(seed)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        totals, seen = np.zeros(3), 0
        try:
            order = rng.permutation(data.train.shape[0])
            for start in range(0, len(order), config["batch_size"]):
                check_limits(train_deadline, max_bytes, device)
                indices = order[start:start+config["batch_size"]]
                x, y = tensors(data, indices, device)
                optimizer.zero_grad(set_to_none=True)
                losses = model.loss(x, y)
                require(torch.isfinite(losses[0]).item(), "Nonfinite training loss")
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizer.step()
                totals += np.array([v.item() for v in losses]) * len(indices)
                seen += len(indices)
            val = validate_epoch(model, data, config["batch_size"], device, train_deadline, max_bytes, seed)
        except ResourceLimit as exc:
            stop_reason = str(exc)
            break
        row = {"epoch": epoch, "train_negative_elbo": float(totals[0]/seen),
               "train_reconstruction_nll": float(totals[1]/seen), "train_kl": float(totals[2]/seen),
               "validation_negative_elbo": val[0], "validation_reconstruction_nll": val[1],
               "validation_kl": val[2], "elapsed_seconds": time.monotonic()-started}
        history.append(row)
        save(out / "history.json", history)
        if val[0] < best:
            best, best_epoch, stale = val[0], epoch, 0
            save_checkpoint(out / "best.pt", model, optimizer, config, data, epoch, seed, job["plan_sha256"])
        else:
            stale += 1
        save_checkpoint(out / "last.pt", model, optimizer, config, data, epoch, seed, job["plan_sha256"])
        print(f"{config['name']} epoch {epoch}: train={row['train_negative_elbo']:.3f}, val={val[0]:.3f}", flush=True)
        if stale >= config["patience"]:
            stop_reason = "early_stopping"
            break
    require(best_epoch is not None, "No full training/validation epoch completed within resource limits; no usable checkpoint")
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    generated = generate(model, data, config, out, seed, device, deadline, max_bytes)
    if plan["task"] == "representation_learning":
        representations(model, data, config["batch_size"], out, device, deadline, max_bytes)
    check_limits(deadline, max_bytes, device)
    plots(out, history, generated, data.metadata["kind"] == "image")
    result = {"status": "pilot_trained" if job["pilot"] else "trained", "model": config["name"],
              "device": device, "best_epoch": best_epoch, "best_validation_negative_elbo": best,
              "epochs_completed": len(history), "stop_reason": stop_reason,
              "generated_samples": len(generated["samples"]), "elapsed_seconds": time.monotonic()-started,
              "train_observations": data.train.shape[0], "validation_observations": data.val.shape[0],
              "test_used": False, "quality_evaluation_complete": False,
              "trainer_contract": TRAINERS[config["name"]], "common_contract": COMMON,
              "limitations": ["Training diagnostics are not evidence of synthetic-data quality or downstream utility.",
                              "Pilot runs override budgets/subset sizes and are not final experiments." if job["pilot"] else "Downstream evaluation is pending.",
                              "Seeds are recorded; numerical identity across devices/library versions is not guaranteed."]}
    save(out / "result.json", result)
    return result


def main():
    import sys
    job_path = Path(sys.argv[1]).resolve()
    out = job_path.parent
    try:
        train_job(json.loads(job_path.read_text()), out)
    except Exception as exc:
        import traceback
        save(out / "result.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                                   "quality_evaluation_complete": False})
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
