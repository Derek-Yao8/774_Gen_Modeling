"""Command-line profiling, with machine-readable failures."""

import argparse
import json
from pathlib import Path

from .data import InputError, load_dataset
from .profile import profile_dataset, render_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect a dataset for objective-dependent generative modeling.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--objective", required=True, help="User objective, preserved for the future planner")
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--label-column")
    parser.add_argument("--id-column")
    parser.add_argument("--orientation", choices=["rows", "columns"])
    parser.add_argument("--split", choices=["train", "val", "test"])
    args = parser.parse_args(argv)
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("Output must be a new or empty directory to avoid mixing runs.")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        data = load_dataset(args.dataset, label_column=args.label_column, id_column=args.id_column,
                            orientation=args.orientation, split=args.split)
        profile = profile_dataset(data, args.objective)
        (args.out / "profile.json").write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
        (args.out / "profile.md").write_text(render_summary(profile), encoding="utf-8")
    except (InputError, ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        failure = {"status": "input_error", "error_type": type(exc).__name__, "message": str(exc)}
        (args.out / "validation_error.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(f"Input could not be profiled: {exc}")
        return 2
    print(f"Profile written to {args.out.resolve()}")
    if profile["requires_clarification"]:
        print("Profile includes unresolved input questions; resolve these before downstream analysis.")
    return 0
