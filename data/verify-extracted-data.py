"""Verify batch_*.npz files produced by process-data.py.

Expected shapes per batch file:
  X: dtype=object, shape (batch_size,)   -- one entry per folder
     X[i]: ndarray, shape (n_i, 64)      -- n_i frames, 64 features each
           columns 0:63 = flattened (x, y, z) for 21 landmarks
           column 63    = handedness (0.0=Left, 1.0=Right, -1.0=no hand detected)
  y: shape (batch_size,), string labels, each in TARGETS
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = [
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Up",
    "Sliding Two Fingers Down",
    "No gesture",
    "Doing other things",
]

FEATURES_PER_FRAME = 64
LANDMARK_COLS = 63
BATCH_SIZE = 32

COORD_XY_RANGE = (-0.5, 1.5)   # normalized image coords, with slack for off-frame landmarks
COORD_Z_RANGE = (-1.0, 1.0)    # relative depth
VALID_HANDEDNESS = {-1.0, 0.0, 1.0}


def expected_batch_sizes(metadata_path, split, batch_size=BATCH_SIZE):
    csv_name = {"train": "train.csv", "validation": "validation.csv", "test": "test-answers.csv"}[split]
    sep = ";"
    df = pd.read_csv(metadata_path / csv_name, sep=sep, header=None, names=["id", "label"])
    target_df = df[df["label"].isin(TARGETS)]
    n = len(target_df)
    n_batches = (n + batch_size - 1) // batch_size
    sizes = {}
    for b in range(n_batches):
        start = b * batch_size
        sizes[b] = min(batch_size, n - start)
    return sizes


def validate_batch(path, batch_num, expected_size):
    result = {
        "batch": batch_num,
        "file": str(path),
        "failures": [],
        "warnings": [],
        "stats": {},
    }

    try:
        with np.load(path, allow_pickle=True) as npz:
            if "X" not in npz or "y" not in npz:
                result["failures"].append(f"missing key(s), found: {list(npz.keys())}")
                return result
            X, y = npz["X"], npz["y"]
    except Exception as e:
        result["failures"].append(f"failed to load file: {e}")
        return result

    # top-level shape
    if X.dtype != object:
        result["failures"].append(f"X.dtype is {X.dtype}, expected object")
    if X.ndim != 1:
        result["failures"].append(f"X.ndim is {X.ndim}, expected 1")
    if y.ndim != 1:
        result["failures"].append(f"y.ndim is {y.ndim}, expected 1")
    if len(X) != len(y):
        result["failures"].append(f"len(X)={len(X)} != len(y)={len(y)}")
    if y.dtype.kind not in ("U", "S"):
        result["failures"].append(f"y.dtype is {y.dtype}, expected string")

    if expected_size is not None and len(y) != expected_size:
        result["failures"].append(f"batch size {len(y)} != expected {expected_size}")

    # labels
    labels = [str(v) for v in y]
    bad_labels = sorted(set(l for l in labels if l not in TARGETS))
    if bad_labels:
        result["failures"].append(f"labels outside TARGETS: {bad_labels}")

    # per-folder checks
    total_frames = 0
    sentinel_frames = 0
    nan_inf_folders = []
    bad_shape_folders = []
    bad_handedness_folders = []
    out_of_range_folders = []
    empty_folders = []

    for i, folder in enumerate(X):
        arr = np.asarray(folder)
        if arr.ndim != 2 or arr.shape[1] != FEATURES_PER_FRAME:
            bad_shape_folders.append((i, arr.shape))
            continue

        n_i = arr.shape[0]
        if n_i == 0:
            empty_folders.append(i)
            continue

        total_frames += n_i

        if not np.isfinite(arr).all():
            nan_inf_folders.append(i)

        landmarks = arr[:, :LANDMARK_COLS]
        handedness = arr[:, LANDMARK_COLS]

        bad_hand = ~np.isin(handedness, list(VALID_HANDEDNESS))
        if bad_hand.any():
            bad_handedness_folders.append(i)

        is_sentinel = (handedness == -1.0) & np.all(landmarks == 0.0, axis=1)
        sentinel_frames += int(is_sentinel.sum())

        real_frames = landmarks[~is_sentinel]
        if real_frames.size:
            xy = real_frames.reshape(-1, 21, 3)[:, :, :2]
            z = real_frames.reshape(-1, 21, 3)[:, :, 2]
            if (xy < COORD_XY_RANGE[0]).any() or (xy > COORD_XY_RANGE[1]).any():
                out_of_range_folders.append(i)
            elif (z < COORD_Z_RANGE[0]).any() or (z > COORD_Z_RANGE[1]).any():
                out_of_range_folders.append(i)

    if bad_shape_folders:
        result["failures"].append(f"folders with wrong shape (expected (n,{FEATURES_PER_FRAME})): {bad_shape_folders[:10]}")
    if empty_folders:
        result["failures"].append(f"folders with zero frames: {empty_folders[:20]}")
    if nan_inf_folders:
        result["failures"].append(f"folders with NaN/Inf values: {nan_inf_folders[:20]}")
    if bad_handedness_folders:
        result["failures"].append(f"folders with handedness outside {{-1,0,1}}: {bad_handedness_folders[:20]}")
    if out_of_range_folders:
        result["warnings"].append(f"folders with landmark coords outside plausible range: {out_of_range_folders[:20]}")

    sentinel_rate = sentinel_frames / total_frames if total_frames else 0.0
    if sentinel_rate > 0.5:
        result["warnings"].append(f"high no-hand-detected rate: {sentinel_rate:.1%} of frames")

    result["stats"] = {
        "num_folders": len(X),
        "total_frames": total_frames,
        "avg_frames_per_folder": total_frames / len(X) if len(X) else 0,
        "sentinel_frame_rate": sentinel_rate,
        "label_counts": {label: labels.count(label) for label in TARGETS if labels.count(label)},
    }
    result["passed"] = len(result["failures"]) == 0
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify process-data.py output batches")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Directory containing batch_*.npz (default: ../../data-repo/target relative to this file)")
    parser.add_argument("--metadata-dir", type=Path, default=None,
                         help="Directory containing train/validation/test-answers.csv")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train",
                         help="Which metadata CSV to use for expected batch-size checks")
    parser.add_argument("--report", type=Path, default=None,
                         help="Path to write JSON report (default: <output-dir>/verification_report.json)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    repo_path = project_root.parent / "data-repo"
    output_dir = args.output_dir or (repo_path / "target")
    metadata_dir = args.metadata_dir or (repo_path / "metadata")
    report_path = args.report or (output_dir / "verification_report.json")

    batch_files = sorted(
        output_dir.glob("batch_*.npz"),
        key=lambda p: int(p.stem.split("_")[1])
    )
    if not batch_files:
        print(f"No batch_*.npz files found in {output_dir}")
        return

    try:
        sizes = expected_batch_sizes(metadata_dir, args.split)
    except FileNotFoundError as e:
        print(f"Could not load metadata for expected batch sizes ({e}); skipping size checks")
        sizes = {}

    checkpoint_file = output_dir / "checkpoint.json"
    checkpoint_last = None
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint_last = json.load(f).get("last_completed_batch")

    results = []
    for path in batch_files:
        batch_num = int(path.stem.split("_")[1])
        expected_size = sizes.get(batch_num)
        results.append(validate_batch(path, batch_num, expected_size))

    max_batch_found = max(r["batch"] for r in results)
    if checkpoint_last is not None and checkpoint_last != max_batch_found:
        print(f"NOTE: checkpoint.json says last_completed_batch={checkpoint_last}, "
              f"but highest batch file present is batch_{max_batch_found}")

    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    print(f"Verified {len(results)} batch file(s) in {output_dir}")
    print(f"  passed: {len(passed)}")
    print(f"  failed: {len(failed)}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        warn_suffix = f", {len(r['warnings'])} warning(s)" if r["warnings"] else ""
        print(f"  batch_{r['batch']}: {status}{warn_suffix}  "
              f"({r['stats'].get('num_folders', '?')} folders, "
              f"{r['stats'].get('total_frames', '?')} frames, "
              f"sentinel_rate={r['stats'].get('sentinel_frame_rate', 0):.1%})")
        for f in r["failures"]:
            print(f"      FAILURE: {f}")
        for w in r["warnings"]:
            print(f"      WARNING: {w}")

    report = {
        "output_dir": str(output_dir),
        "checkpoint_last_completed_batch": checkpoint_last,
        "num_batches_checked": len(results),
        "num_passed": len(passed),
        "num_failed": len(failed),
        "batches": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {report_path}")


if __name__ == "__main__":
    main()
