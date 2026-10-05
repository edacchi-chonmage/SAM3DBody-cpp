#!/usr/bin/env python3
"""Contact sheet: video frames next to a BVH stick figure at the same time.

Usage:
    python3 tools/bvh_side_by_side.py VIDEO BVH -o out.jpg \
        [--times 1,4,7,10,13,16,19] [--views front,side]

One row per time t (seconds): [video frame | front (X/Y) | side (Z/Y)].
BVH frame = round(t / FrameTime), clamped.  Y is up; units are whatever the
BVH uses.  Axis limits are fixed over the whole clip (root range + body size)
so motion is comparable across rows.  Face/finger joints are not drawn.
"""
import argparse
import math
import sys

import cv2
import numpy as np

SKIP = ("finger", "thumb", "index", "middle", "ring", "pinky", "jaw", "eye", "tongue",
        "metacarpal", "levator", "oris", "special", "temporalis", "oculi", "risorius",
        "orbicularis", "toe2", "toe3", "toe4", "toe5")
H = 360


def parse_bvh(path):
    toks = open(path).read().split("MOTION")[0].split()
    names, parent, offs, chans, stack, pend = [], [], [], [], [], None
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            pend = toks[i + 1]; i += 2; continue
        if t == "End":
            pend = ""; i += 2; continue
        if t == "{":
            names.append(pend or ""); parent.append(stack[-1] if stack else -1)
            offs.append(np.zeros(3)); chans.append([]); stack.append(len(names) - 1)
        elif t == "}":
            stack.pop()
        elif t == "OFFSET":
            offs[stack[-1]] = np.array(toks[i + 1:i + 4], float); i += 3
        elif t == "CHANNELS":
            n = int(toks[i + 1]); chans[stack[-1]] = toks[i + 2:i + 2 + n]; i += 1 + n
        i += 1
    lines = open(path).read().split("MOTION")[1].strip().splitlines()
    n = int(lines[0].split(":")[1]); ft = float(lines[1].split(":")[1])
    data = np.array([l.split() for l in lines[2:2 + n]], float)
    return names, parent, offs, chans, data, ft


def rot(axis, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    if axis == "X": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk(names, parent, offs, chans, row):
    pos = np.zeros((len(names), 3)); R = [None] * len(names); k = 0
    for j in range(len(names)):
        loc, m = offs[j].copy(), np.eye(3)
        for ch in chans[j]:
            v = row[k]; k += 1
            if ch.endswith("position"):
                loc[("XYZ".index(ch[0]))] += v   # position channels add to the OFFSET
            else:
                m = m @ rot(ch[0], v)          # channel order = multiplication order
        if parent[j] < 0:
            pos[j], R[j] = loc, m
        else:
            pos[j] = pos[parent[j]] + R[parent[j]] @ loc
            R[j] = R[parent[j]] @ m
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("bvh")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--times", default="1,4,7,10,13,16,19")
    ap.add_argument("--views", default="front,side")
    ap.add_argument("--follow", action="store_true",
                    help="centre each panel on that frame's root and zoom to the body (pose check); "
                         "default keeps whole-clip limits so travel is visible")
    a = ap.parse_args()
    times = [float(x) for x in a.times.split(",")]
    views = a.views.split(",")

    names, parent, offs, chans, data, ft = parse_bvh(a.bvh)
    keep = [bool(n) and not n.startswith("__") and not any(s in n.lower() for s in SKIP)
            for n in names]
    keep[0] = True
    anc = []  # nearest kept ancestor
    for j, p in enumerate(parent):
        while p >= 0 and not keep[p]: p = parent[p]
        anc.append(p)
    edges = [(anc[j], j) for j in range(len(names)) if keep[j] and anc[j] >= 0]

    # whole-clip extent: FK on a subsample of frames
    idx = np.unique(np.linspace(0, len(data) - 1, min(len(data), 120)).astype(int))
    allp = np.concatenate([fk(names, parent, offs, chans, data[i])[keep] for i in idx])
    lo, hi = allp.min(0), allp.max(0)
    ground = lo[1]
    span = max(hi[1] - lo[1], hi[0] - lo[0], hi[2] - lo[2]) * 1.1
    cen = (lo + hi) / 2
    pad = 0.05 * span
    # y is the same scale in both views; horizontal axis is centred on the clip
    body = (hi[1] - lo[1]) if not a.follow else 0
    def to_px(p, hax, c=None, sp=None):
        c = cen if c is None else c
        sp = span if sp is None else sp
        s = (H - 20) / sp
        return int(H / 2 + (p[hax] - c[hax]) * s), int(H / 2 - (p[1] - c[1]) * s)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        if not ok: fr = np.zeros((H, int(H * 16 / 9), 3), np.uint8)
        fr = cv2.resize(fr, (int(fr.shape[1] * H / fr.shape[0]), H))
        bf = min(max(int(round(t / ft)), 0), len(data) - 1)
        p = fk(names, parent, offs, chans, data[bf])
        c, sp, gnd = None, None, ground
        if a.follow:
            pk = p[keep]
            c = (pk.min(0) + pk.max(0)) / 2
            sp = max(pk.max(0)[1] - pk.min(0)[1], 120) * 1.3
            gnd = pk.min(0)[1]
        panels = [fr]
        for v in views:
            hax = 0 if v == "front" else 2
            img = np.full((H, H, 3), 255, np.uint8)
            gy = to_px(np.array([0, gnd, 0]), 0, c, sp)[1]
            cv2.line(img, (0, gy), (H, gy), (170, 170, 170), 1)
            for u, w in edges:
                col = (0, 0, 200) if "r" == names[w][0] and names[w] != "rCollar" and "Foot" in names[w] + names[u] else (60, 60, 60)
                cv2.line(img, to_px(p[u], hax, c, sp), to_px(p[w], hax, c, sp), col, 2, cv2.LINE_AA)
            cv2.putText(img, v, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(img)
        row = np.hstack(panels)
        cv2.putText(row, f"t={t:g}s bvh#{bf}", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)
        rows.append(row)
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    sheet = np.vstack(rows)
    if sheet.shape[1] > 1600:
        s = 1600 / sheet.shape[1]
        sheet = cv2.resize(sheet, (1600, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {a.out} {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
