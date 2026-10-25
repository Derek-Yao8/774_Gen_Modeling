"""Execute a saved, revalidated plan in supervised per-model worker processes."""
import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .capabilities import TRAINERS
from .planning import require, save, validate


def supervise(command, cwd, env, log_path, wall_seconds, max_bytes):
    import psutil
    started = time.monotonic()
    peak_rss = 0
    limit = None
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            process = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    rss = process.memory_info().rss
                    peak_rss = max(peak_rss, rss)
                except psutil.NoSuchProcess:
                    break
                if rss > max_bytes:
                    limit = "rss_memory_limit"
                elif time.monotonic() - started > wall_seconds:
                    limit = "wall_time_limit"
                if limit:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                    break
                time.sleep(0.2)
            code = child.wait()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    return {"exit_code": code, "limit_triggered": limit, "peak_worker_rss_bytes": peak_rss,
            "wall_seconds": time.monotonic() - started}


def run(plan_path, out, *, dataset=None, device="auto", pilot=False, max_memory_gib=6):
    require(device in ("auto", "cpu", "mps", "cuda"), "Invalid device")
    require(type(max_memory_gib) in (int, float) and 0.5 <= max_memory_gib <= 64, "Memory allowance must be 0.5..64 GiB")
    plan_path = Path(plan_path).resolve()
    raw = plan_path.read_bytes()
    envelope = json.loads(raw)
    require(envelope.get("status") == "ready", "Only ready plans can execute")
    profile = json.loads((plan_path.parent / "input_profile.json").read_text())
    plan = validate({"status": "ready", "explanation": envelope["explanation"], "questions": [],
                     "plan_json": json.dumps(envelope["plan"])}, profile, envelope["budget"], legacy_catalog=envelope.get("catalog_version") != 2)
    unsupported = [m["name"] for m in plan["models"] if not TRAINERS[m["name"]]["implemented"]]
    require(not unsupported, f"Unavailable trainers: {unsupported}; no model substitution is performed")
    source = Path(dataset) if dataset else Path(profile["provenance"]["sources"][0]["path"])
    if not dataset and profile["provenance"]["format"] == "10x_mtx":
        source = source.parent
    require(source.exists(), "Dataset not found; use --dataset to relocate matching input data")
    out = Path(out).resolve()
    require(not out.exists(), "Choose a new execution output directory")
    out.mkdir(parents=True)
    (out / "source_modeling_plan.json").write_bytes(raw)
    save(out / "input_profile.json", profile)
    sha = hashlib.sha256(raw).hexdigest()
    provenance = {"plan_sha256": sha, "dataset": str(source.resolve()), "pilot": pilot,
                  "device_requested": device, "max_memory_gib": max_memory_gib,
                  "python": platform.python_version(),
                  "software": {name: version(name) for name in ("torch", "numpy", "scipy", "scikit-learn", "matplotlib", "psutil")},
                  "source_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    save(out / "provenance.json", provenance)
    effective = deepcopy(plan)
    overrides = []
    for config in effective["models"]:
        if pilot:
            for key, ceiling in (("epochs", 2), ("patience", 2), ("generate_samples", 32)):
                old = config[key]
                config[key] = min(old, ceiling)
                if old != config[key]:
                    overrides.append({"model": config["name"], "field": key, "planned": old, "pilot": config[key]})
    save(out / "effective_plan.json", {"plan": effective, "pilot": pilot, "overrides": overrides,
                                       "pilot_subset_caps": {"train": 512, "validation": 128} if pilot else None})
    records = []
    root = str(Path(__file__).resolve().parent.parent)
    for config in effective["models"]:
        model_out = out / config["name"]
        model_out.mkdir()
        wall = min(config["max_minutes"]*60, 180) if pilot else config["max_minutes"]*60
        job = {"dataset": str(source.resolve()), "profile": profile, "plan": effective, "config": config,
               "plan_sha256": sha, "pilot": pilot, "device": device,
               "max_bytes": int(max_memory_gib * 2**30), "wall_seconds": wall}
        save(model_out / "job.json", job)
        env = os.environ.copy()
        env.update(PYTHONPATH=root, PYTHONNOUSERSITE="1", MPLCONFIGDIR=str(model_out / "matplotlib_cache"),
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        # Do not silently route unsupported MPS operations to CPU mid-training.
        env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        monitor = supervise([sys.executable, "-m", "genmodel_agent.trainer", str(model_out / "job.json")],
                            root, env, model_out / "training.log", wall + 15, job["max_bytes"])
        result_path = model_out / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "failed", "error": "Worker did not produce a result"}
        if monitor["exit_code"] != 0 or monitor["limit_triggered"]:
            result["status"] = "failed"
            result.setdefault("error", monitor["limit_triggered"] or f"Worker exit {monitor['exit_code']}")
        result.update(supervision=monitor, model=config["name"])
        save(result_path, result)
        records.append(result)
        save(out / "execution.json", {"status": "running", "models": records})
    success = all(r["status"] in ("trained", "pilot_trained") for r in records)
    outcome = {"status": "completed" if success else "failed_or_partial", "pilot": pilot, "models": records,
               "training_executed": True, "quality_evaluation_complete": False, "test_used": False,
               "plan_sha256": sha, "overrides": overrides}
    save(out / "execution.json", outcome)
    lines = ["# Execution report", "", f"Status: **{outcome['status']}**", "",
             "**Pilot run: reduced subsets and budgets; not a final experiment.**" if pilot else "Execution of the saved modeling plan.", "",
             "Test data were reserved. Synthetic-data quality and downstream utility have not been evaluated.", ""]
    for result in records:
        lines += [f"## {result['model']}", "", f"- Status: {result['status']}"]
        for key in ("device", "train_observations", "validation_observations", "epochs_completed", "best_epoch", "best_validation_negative_elbo", "best_validation_nll", "best_validation_feature_wasserstein", "generated_samples", "stop_reason", "error"):
            if key in result:
                lines.append(f"- {key}: {result[key]}")
        lines += [f"- Peak worker RSS: {result['supervision']['peak_worker_rss_bytes']/2**30:.2f} GiB",
                  f"- Wall time: {result['supervision']['wall_seconds']:.1f} seconds", ""]
    lines += ["## Review artifacts", "", "Each model folder contains the exact job, preprocessing/split records, training log, and result. Successful runs also contain best/last checkpoints, epoch history, a learning curve and generated.npz. Representation objectives additionally save train/validation posterior means. These artifacts do not establish that generated data should be used downstream.", ""]
    (out / "execution_report.md").write_text("\n".join(lines), encoding="utf-8")
    return outcome


def main():
    parser = argparse.ArgumentParser(description="Execute an existing modeling plan without new AI calls")
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Relocated input with identical source hashes")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--pilot", action="store_true", help="Explicit 512/128-observation, two-epoch, 32-sample pilot caps")
    parser.add_argument("--max-memory-gib", type=float, default=6)
    args = parser.parse_args()
    try:
        result = run(args.plan, args.out, dataset=args.dataset, device=args.device,
                     pilot=args.pilot, max_memory_gib=args.max_memory_gib)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(2, f"Execution refused: {exc}\n")
    print(f"{result['status']}: {args.out.resolve() / 'execution_report.md'}")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
