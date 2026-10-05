#!/usr/bin/env python3
"""Synthetic board pose for testing: under the feet, yaw from rFoot->lFoot, fake roll at frames 300-330.

Usage: gen_test_board_csv.py PERSON.bvh OUT.csv
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_side_by_side as bss

names, parent, offs, chans, data, ft = bss.parse_bvh(sys.argv[1])
l, r = names.index("lFoot"), names.index("rFoot")
rows = ["frame,tx,ty,tz,rz,ry,rx,source"]
for i, row in enumerate(data):
    p = bss.fk(names, parent, offs, chans, row)
    c = (p[l] + p[r]) / 2 - [0, 8, 0]
    d = p[l] - p[r]
    # R = Rz Ry Rx; with only Ry, +X maps to (cos, 0, -sin): yaw so +X follows d (xz plane)
    ry = math.degrees(math.atan2(-d[2], d[0]))
    rx = 360.0 * (i - 300) / 30 if 300 <= i <= 330 else 0.0
    rows.append(f"{i},{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},0,{ry:.4f},{rx:.4f},feet")
open(sys.argv[2], "w").write("\n".join(rows) + "\n")
