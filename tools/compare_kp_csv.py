#!/usr/bin/env python3
"""Compare two 3D keypoint CSVs written by fast_sam_3dbody_run --out.

Usage: compare_kp_csv.py A.csv B.csv [--skeleton 0] [--root-relative]

CSV header: frame,skeleton_id,<name>_x,<name>_y,<name>_z (metres).
Rows are matched by (frame, skeleton_id). Distances are printed in mm.
--root-relative subtracts joint 'pelvis' (or the mean of all joints) and
prints raw and root-relative results.
"""
import argparse
import csv
import numpy as np


def load(path, skel):
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        names = [h[:-2] for h in head[2::3]]
        rows = {}
        for row in r:
            if skel is not None and int(row[1]) != skel:
                continue
            rows[(int(row[0]), int(row[1]))] = np.array(row[2:], float).reshape(-1, 3)
    return names, rows


def report(title, A, B, keys, names):
    d = np.stack([np.linalg.norm(A[k] - B[k], axis=1) for k in keys]) * 1000  # F x J
    print(f"== {title} ==")
    for k, row in zip(keys, d):
        print(f"frame {k[0]} skel {k[1]}: max {row.max():.3f} mm  mean {row.mean():.3f} mm")
    print(f"overall: max {d.max():.3f}  mean {d.mean():.3f}  median {np.median(d):.3f} mm")
    worst = np.argsort(-d.mean(axis=0))[:3]
    print("worst joints (mean mm): " + ", ".join(f"{names[j]} {d[:, j].mean():.3f}" for j in worst))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--skeleton", type=int, default=0)
    p.add_argument("--root-relative", action="store_true")
    a = p.parse_args()
    names, A = load(a.a, a.skeleton)
    _, B = load(a.b, a.skeleton)
    keys = sorted(set(A) & set(B))
    print(f"matched frames: {len(keys)}")
    if not keys:
        return
    report("raw", A, B, keys, names)
    if a.root_relative:
        def rel(X):
            return {k: v - (v[names.index("pelvis")] if "pelvis" in names else v.mean(0)) for k, v in X.items()}
        report("root-relative", rel(A), rel(B), keys, names)


main()
