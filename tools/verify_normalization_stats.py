#!/usr/bin/env python3
"""CPU-only, bounded-memory audit of train_unet_v2.py normalization.

The ordinary experiment uses random simulation splitting, no temporal split,
seed=0, train_frac=0.9, and frames 0..20. Statistics precede the train/val split.
No training module is imported and no checkpoint is deserialized.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import platform
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


def check_training_source(path):
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    precise_assignment = ast.parse("mean, std = compute_normalization_stats(train_raw)").body[0]
    if any(ast.dump(node) == ast.dump(precise_assignment) for node in main.body):
        return source
    for name, expected in {
        "mean": "train_raw.reshape(-1, C).mean(axis=0).astype(np.float64)",
        "std": "train_raw.reshape(-1, C).std(axis=0).astype(np.float64)",
    }.items():
        candidates = [node for node in main.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
        if len(candidates) != 1 or ast.dump(candidates[0].value) != ast.dump(ast.parse(expected, mode="eval").body):
            raise ValueError(f"Training expression for {name} has changed; review the verifier first.")
    return source


def select_training_data(handle, args):
    """Match sorted keys, float32 h-byte dedup, first representatives and RNG order."""
    keys = sorted(handle.keys())
    if not keys:
        raise ValueError("HDF5 contains no simulations.")
    seen = {}
    representative_keys = []
    gids = []
    for key in keys:
        data = handle[key]["data"]
        if data.ndim != 4 or data.shape[-1] != 3:
            raise ValueError(f"{key}/data must have shape [T,H,W,3]; got {data.shape}.")
        if args.t_start >= data.shape[0]:
            raise ValueError(f"t_start={args.t_start} is outside {key}/data.")
        # Reading only h is equivalent to rep_fields[..., 0].tobytes().
        initial_h = np.asarray(data[args.t_start, :, :, 0], dtype=np.float32)
        signature = initial_h.tobytes()
        if signature not in seen:
            seen[signature] = len(seen)
            representative_keys.append(key)
        gids.append(seen[signature])
    unique_count = len(representative_keys)
    order = np.random.default_rng(args.seed).permutation(unique_count)
    n_train = max(1, int(round(args.train_frac * unique_count)))
    train_groups = order[:n_train]
    test_groups = order[n_train:]
    frame_indices = list(range(args.t_start, args.t_end + 1, args.t_step))
    training_keys = [representative_keys[int(g)] for g in train_groups]
    shape = tuple(handle[training_keys[0]]["data"].shape[1:])
    for key in training_keys:
        ds = handle[key]["data"]
        if tuple(ds.shape[1:]) != shape or frame_indices[-1] >= ds.shape[0]:
            raise ValueError(f"Incompatible shape or frame interval: {key}/data {ds.shape}.")
    return {
        "n_simulations": len(keys), "n_unique": unique_count,
        "n_train_groups": len(train_groups), "n_test_groups": len(test_groups),
        "train_group_ids_ordered": train_groups.tolist(),
        "test_group_ids_ordered": test_groups.tolist(),
        "training_keys_ordered": training_keys,
        "representative_keys_ordered": representative_keys,
        "frame_indices": frame_indices, "spatial_channel_shape": list(shape),
        "pixels_per_channel": len(training_keys) * len(frame_indices) * shape[0] * shape[1],
    }, np.asarray(gids)


def iter_chunks(handle, selection, chunk_rows):
    height, width, _ = selection["spatial_channel_shape"]
    # HDF5 hyperslabs keep even a single large frame bounded in memory.
    row_step = max(1, chunk_rows // width)
    for key in selection["training_keys_ordered"]:
        data = handle[key]["data"]
        for frame in selection["frame_indices"]:
            for row in range(0, height, row_step):
                block = np.asarray(data[frame, row:row + row_step, :, :], dtype=np.float32)
                flat = np.ascontiguousarray(block).reshape(-1, 3)
                for start in range(0, len(flat), chunk_rows):
                    yield flat[start:start + chunk_rows]


def accumulate32(total, values, scratch):
    # A sum of chunk sums would change floating-point rounding. Carry the last
    # float32 accumulator into the next sequential accumulation instead.
    work = scratch[:len(values) + 1]
    work[0] = total
    work[1:] = values
    np.add.accumulate(work, axis=0, dtype=np.float32, out=work)
    total[:] = work[-1]


def compute_statistics(chunk_factory, n_rows, chunk_rows, progress=False):
    scratch = np.empty((chunk_rows + 1, 3), dtype=np.float32)
    sum32 = np.zeros(3, dtype=np.float32)
    sum64 = np.zeros(3, dtype=np.float64)
    count = 0
    for block in chunk_factory():
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite values in training data; statistics are not trustworthy.")
        accumulate32(sum32, block, scratch)
        sum64 += block.sum(axis=0, dtype=np.float64)
        count += len(block)
    if count != n_rows or count == 0:
        raise ValueError(f"Pixel count mismatch: read {count}, expected {n_rows}.")
    mean32 = sum32.copy()
    np.true_divide(mean32, np.intp(n_rows), out=mean32, casting="unsafe")
    mean64 = sum64 / n_rows
    if progress:
        print(f"Pass 1/2 complete: {count:,} pixels/channel", flush=True)
    var32 = np.zeros(3, dtype=np.float32)
    squared_deviations64 = np.zeros(3, dtype=np.float64)
    count2 = 0
    for block in chunk_factory():
        # np.std uses float32 subtraction/squaring and its float32 mean.
        deviations32 = np.subtract(block, mean32, dtype=np.float32)
        np.square(deviations32, out=deviations32)
        accumulate32(var32, deviations32, scratch)
        deviations64 = block.astype(np.float64) - mean64
        squared_deviations64 += np.einsum("ij,ij->j", deviations64, deviations64)
        count2 += len(block)
    if count2 != n_rows:
        raise ValueError("Training dataset changed between passes.")
    np.true_divide(var32, np.intp(n_rows), out=var32, casting="unsafe")
    np.sqrt(var32, out=var32)
    std64 = np.sqrt(squared_deviations64 / n_rows)
    if progress:
        print("Pass 2/2 complete", flush=True)
    return mean32.astype(np.float64), var32.astype(np.float64), mean64, std64


def validate_replay():
    """Fail closed if this NumPy build's axis=0 reduction differs from replay."""
    rng = np.random.default_rng(4471)
    for size in (17, 4099, 131073):
        x = rng.normal(size=(size, 3)).astype(np.float32)
        x[:, 0] += 2
        for chunk_rows in (113, 4096):
            factory = lambda: (x[i:i + chunk_rows] for i in range(0, len(x), chunk_rows))
            m32, s32, m64, s64 = compute_statistics(factory, len(x), chunk_rows)
            if not (np.array_equal(m32, x.mean(axis=0).astype(np.float64))
                    and np.array_equal(s32, x.std(axis=0).astype(np.float64))):
                raise RuntimeError("Float32 replay is not bit-identical on this NumPy build.")
            np.testing.assert_allclose(m64, x.mean(axis=0, dtype=np.float64), rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(s64, x.std(axis=0, dtype=np.float64), rtol=1e-12, atol=1e-13)


def self_test(source):
    """Compare the verifier with actual pre-training main() statements on a tiny HDF5."""
    validate_replay()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    names = ("load_swe_representative", "unique_group_ids", "load_dam_radius", "load_sim_params", "load_swe_timeseries")
    if "compute_normalization_stats" in functions:
        names += ("compute_normalization_stats",)
    reference = {"np": np, "h5py": h5py, "re": __import__("re")}
    exec(compile(ast.Module(body=[functions[name] for name in names], type_ignores=[]), "training_reference", "exec"), reference)
    main_body = functions["main"].body
    start = next(i for i, node in enumerate(main_body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == "rep_fields" for e in t.elts) for t in node.targets))
    end = next(i for i, node in enumerate(main_body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "std"
                       for target in node.targets for t in ast.walk(target)))
    pre_training = ast.Module(body=main_body[start:end + 1], type_ignores=[])
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="comp4471-stats-test-") as temp:
        path = Path(temp) / "synthetic.h5"
        with h5py.File(path, "w") as f:
            fields = rng.normal(size=(13, 3, 12, 12, 3)).astype(np.float32)
            fields[11, 0, :, :, 0] = fields[1, 0, :, :, 0]
            fields[12, 0, :, :, 0] = fields[2, 0, :, :, 0]
            for i, field in enumerate(fields):
                group = f.create_group(f"{i:04}")
                group.create_dataset("data", data=field)
                group.attrs["config"] = f"dam_radius: {0.3 + i * .02}\ninner_height: 2.0\n"
        for seed, start_frame, step in ((0, 0, 1), (7, 0, 2), (3, 1, 1)):
            args = argparse.Namespace(data_path=str(path), seed=seed, train_frac=.9,
                                      t_start=start_frame, t_end=2, t_step=step,
                                      split_mode="random", temporal_split="none")
            expected = dict(reference, args=args)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(pre_training, "actual_training_main", "exec"), expected)
            with h5py.File(path, "r") as handle:
                selection, gids = select_training_data(handle, args)
                factory = lambda: iter_chunks(handle, selection, 37)
                got = compute_statistics(factory, selection["pixels_per_channel"], 37)
                streamed = np.concatenate(list(factory()))
            np.testing.assert_array_equal(gids, expected["gids"])
            np.testing.assert_array_equal(selection["train_group_ids_ordered"], expected["train_groups"])
            np.testing.assert_array_equal(selection["test_group_ids_ordered"], expected["test_groups"])
            np.testing.assert_array_equal(streamed, expected["train_raw"].reshape(-1, 3))
            precise = "compute_normalization_stats" in functions
            if precise:
                np.testing.assert_allclose(got[2], expected["mean"], rtol=1e-12, atol=1e-13)
                np.testing.assert_allclose(got[3], expected["std"], rtol=1e-12, atol=1e-13)
            else:
                np.testing.assert_array_equal(got[0], expected["mean"])
                np.testing.assert_array_equal(got[1], expected["std"])
    print("PASS: float32 replay bit-identical; float64 reference agrees.")
    print("PASS: actual training split, deduplication, representative order, frame selection and raw pixels match (3 cases).")


def compare_saved_stats(paths, args, selection, m32, s32, m64, s64):
    comparisons = []
    metadata = dict(seed=args.seed, t_start=args.t_start, t_end=args.t_end,
                    t_step=args.t_step, split_mode="random", n_channels=3,
                    n_unique=selection["n_unique"], n_train_groups=selection["n_train_groups"])
    for path in paths:
        record = {"path": str(path.resolve())}
        try:
            old = json.loads(path.read_text(encoding="utf-8-sig"))
            record["metadata_mismatches"] = {k: {"saved": old[k], "requested": v}
                                             for k, v in metadata.items() if k in old and old[k] != v}
            record["missing_metadata"] = [k for k in metadata if k not in old]
            record["same_dataset_proven"] = False
            record["note"] = "Metadata consistency alone does not establish identical HDF5 data or code."
            for key, original, precise in (("mean", m32, m64), ("std", s32, s64)):
                values = np.asarray(old[key], dtype=np.float64)
                if values.shape != (3,) or not np.isfinite(values).all():
                    raise ValueError(f"Saved {key} is not three finite values.")
                record[key] = {"saved": values.tolist(),
                               "abs_diff_vs_float32_replay": np.abs(values - original).tolist(),
                               "abs_diff_vs_float64": np.abs(values - precise).tolist()}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record["error"] = str(exc)
        comparisons.append(record)
    return comparisons


def finite_relative(error, reference):
    return [float(e / abs(r)) if r != 0 else None for e, r in zip(error, reference)]


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-script", type=Path, default=repo / "pdebench/train_unet_v2.py")
    parser.add_argument("--data-path", type=Path, default=repo / "pdebench/data/2D_rdb_NA_NA_1000.h5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=.9)
    parser.add_argument("--t-start", type=int, default=0)
    parser.add_argument("--t-end", type=int, default=20)
    parser.add_argument("--t-step", type=int, default=1)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--stats-json", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, help="Optional JSON report; never overwrites an existing file.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 0 < args.train_frac < 1 or args.t_start < 0 or args.t_end < args.t_start or args.t_step < 1 or args.chunk_rows < 1:
        parser.error("Invalid fraction, frame interval, or chunk size.")
    source = check_training_source(args.training_script)
    if args.self_test:
        self_test(source)
        return 0
    if not args.data_path.is_file():
        print(f"DATA MISSING: {args.data_path}\nPlace the existing HDF5 there, or pass --data-path. Nothing downloaded; no real-data statistics computed.", file=sys.stderr)
        return 2
    if args.output and args.output.exists():
        parser.error(f"Refusing to overwrite existing report: {args.output}")
    validate_replay()
    with h5py.File(args.data_path, "r") as handle:
        selection, _ = select_training_data(handle, args)
        print(f"Selected {selection['n_train_groups']}/{selection['n_unique']} unique simulations, {len(selection['frame_indices'])} frames each, {selection['pixels_per_channel']:,} pixels/channel.", flush=True)
        factory = lambda: iter_chunks(handle, selection, args.chunk_rows)
        m32, s32, m64, s64 = compute_statistics(factory, selection["pixels_per_channel"], args.chunk_rows, progress=True)
    rows = []
    print("\nchannel stat       original_float32       reference_float64          abs_error          rel_error")
    for index, name in enumerate(("h", "u", "v")):
        row = {"channel": name}
        for key, old, accurate in (("mean", m32, m64), ("std", s32, s64)):
            error = np.abs(old - accurate)
            relative = finite_relative(error, accurate)
            row[key] = {"float32": float(old[index]), "float64": float(accurate[index]),
                        "absolute_error": float(error[index]), "relative_error": relative[index]}
            rel_text = "undefined (reference=0)" if relative[index] is None else f"{relative[index]:.9g}"
            print(f"{name:>7} {key:>4} {old[index]:22.12g} {accurate[index]:23.12g} {error[index]:18.9g} {rel_text:>18}")
        row["mean_shift_in_reference_std"] = float(abs(m32[index] - m64[index]) / s64[index]) if s64[index] else None
        row["original_std_over_reference_std"] = float(s32[index] / s64[index]) if s64[index] else None
        rows.append(row)
    stats_paths = sorted(set(args.stats_json) | set(repo.rglob("train_stats.json")))
    report = {
        "data_path": str(args.data_path.resolve()), "data_size_bytes": args.data_path.stat().st_size,
        "training_script": str(args.training_script.resolve()),
        "training_script_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "environment": {"python": sys.version, "numpy": np.__version__, "h5py": h5py.__version__, "platform": platform.platform()},
        "configuration": {"seed": args.seed, "train_frac": args.train_frac, "t_start": args.t_start,
                          "t_end": args.t_end, "t_step": args.t_step, "chunk_rows": args.chunk_rows,
                          "split_mode": "random", "temporal_split": "none", "ddof": 0},
        "scope": "Statistics use the pre-validation train_raw pool, exactly as ordinary training does.",
        "selection": selection, "channels": rows,
        "saved_stats": compare_saved_stats(stats_paths, args, selection, m32, s32, m64, s64),
        "interpretation": "Relative mean error is unstable near zero; inspect mean shift / reference std and std ratios. Numerical differences alone do not prove retraining is necessary. Never replace the normalization of an already-trained checkpoint without retraining."
    }
    print(f"\nExisting train_stats.json files checked: {len(stats_paths)}")
    for saved in report["saved_stats"]:
        print(json.dumps(saved, ensure_ascii=False, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"Report: {args.output.resolve()}")
    print("No training, checkpoint modification, or GPU use performed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, StopIteration) as error:
        print(f"VERIFICATION FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
