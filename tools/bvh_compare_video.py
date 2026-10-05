#!/usr/bin/env python3
"""Video | BVH front | BVH side, one output frame per video frame, written as mp4.

Usage: bvh_compare_video.py VIDEO BVH -o OUT.mp4 [--width 1280] [--crf 26]
Panels use a fixed span for the whole clip and follow the root horizontally;
the ground line stays at the clip-wide minimum height.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_side_by_side as bss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("bvh")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--crf", type=int, default=26)
    a = ap.parse_args()

    names, parent, offs, chans, data, ft = bss.parse_bvh(a.bvh)
    keep = [bool(n) and not n.startswith("__") and not any(s in n.lower() for s in bss.SKIP)
            for n in names]
    keep[0] = True
    anc = []
    for j, p in enumerate(parent):
        while p >= 0 and not keep[p]: p = parent[p]
        anc.append(p)
    edges = [(anc[j], j) for j in range(len(names)) if keep[j] and anc[j] >= 0]

    poses = np.array([bss.fk(names, parent, offs, chans, r) for r in data])
    pk = poses[:, keep]
    body = (pk[:, :, 1].max(1) - pk[:, :, 1].min(1))
    span = max(body.max(), 120) * 1.3
    ground = pk[:, :, 1].min()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    H = int(round(a.width / (vw / vh + 2) / 2)) * 2
    W = a.width - a.width % 2
    VW = W - 2 * H
    if VW % 2: VW -= 1; W -= 1
    s = (H - 20) / span
    gy = int(H * 0.9)

    def px(p, hax, rootx):
        return int(H / 2 + (p[hax] - rootx) * s), int(gy - (p[1] - ground) * s)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ff = subprocess.Popen(
        ["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", str(a.crf), "-preset", "medium", "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE)
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        t = n / fps
        bf = min(max(int(round(t / ft)), 0), len(data) - 1)
        p = poses[bf]
        vid = cv2.resize(fr, (VW, H), interpolation=cv2.INTER_AREA)
        cv2.putText(vid, f"t={t:.2f}s  bvh#{bf}", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        panels = [vid]
        for v in ("front", "side"):
            hax = 0 if v == "front" else 2
            img = np.full((H, H, 3), 255, np.uint8)
            cv2.line(img, (0, gy), (H, gy), (170, 170, 170), 1)
            for u, w in edges:
                col = (0, 0, 200) if "r" == names[w][0] and names[w] != "rCollar" and "Foot" in names[w] + names[u] else (60, 60, 60)
                cv2.line(img, px(p[u], hax, p[0][hax]), px(p[w], hax, p[0][hax]), col, 2, cv2.LINE_AA)
            cv2.putText(img, v, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(img)
        ff.stdin.write(np.hstack(panels).tobytes())
        n += 1
    ff.stdin.close(); ff.wait()
    print(f"{a.out} frames={n} size={os.path.getsize(a.out)/1e6:.2f}MB")


if __name__ == "__main__":
    main()
