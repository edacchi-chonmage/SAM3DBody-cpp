#!/usr/bin/env python3
"""Append a second ROOT 'board' (6 channels) to a person BVH from a per-frame CSV.

Usage: add_board_to_bvh.py PERSON.bvh BOARD.csv -o OUT.bvh
CSV header: frame,tx,ty,tz,rz,ry,rx,source (cm, degrees, BVH channel order Z Y X).
Verifies by re-parsing (multi-root parser): frames, channel count, board FK == CSV.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_side_by_side as bss

BOARD = ("ROOT board\n{\n  OFFSET 0 0 0\n"
         "  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
         "  End Site\n  {\n    OFFSET 40 0 0\n  }\n}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bvh"); ap.add_argument("csv")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    text = open(a.bvh).read()
    head, motion = text.split("MOTION")
    lines = motion.strip().splitlines()
    n = int(lines[0].split(":")[1])
    rows = sorted(csv.DictReader(open(a.csv)), key=lambda r: int(r["frame"]))
    if len(rows) != n:
        sys.exit(f"error: BVH has {n} frames, CSV has {len(rows)}")
    out = [head.rstrip() + "\n" + BOARD + "MOTION", lines[0], lines[1]]
    for line, r in zip(lines[2:2 + n], rows):
        vals = " ".join(r[k] for k in ("tx", "ty", "tz", "rz", "ry", "rx"))
        out.append(line.rstrip() + " " + vals)
    open(a.out, "w").write("\n".join(out) + "\n")

    # verify
    names, parent, offs, chans, data, ft = bss.parse_bvh(a.out)
    p0 = bss.parse_bvh(a.bvh)
    nch = sum(len(c) for c in chans)
    assert len(data) == n, "frame count"
    assert nch == sum(len(c) for c in p0[3]) + 6 == data.shape[1], "channel count"
    b = names.index("board")
    for i in (0, n // 3, 300 if n > 300 else n - 1, n - 1):
        pos = bss.fk(names, parent, offs, chans, data[i])
        want = np.array([float(rows[i][k]) for k in ("tx", "ty", "tz")])
        nose = pos[b + 1]  # End Site follows board
        assert np.abs(pos[b] - want).max() < 1e-3, f"board pos mismatch at {i}"
        print(f"frame {i}: board {pos[b].round(2)} nose-dir {(nose - pos[b]).round(1)}")
    print(f"ok: {a.out} frames={n} channels={nch}")


if __name__ == "__main__":
    main()
