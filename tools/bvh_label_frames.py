#!/usr/bin/env python3
"""Draw each BVH's number on video frames, to see which file is which person.

Usage: bvh_label_frames.py VIDEO OUT.jpg BVH... [--times 1,5,9] [--focal F] [--offsets 0,0,...]
The BVH root/head positions are camera-space cm (X right, Y up, -Z forward), projected with a
pinhole camera at the image centre. --focal defaults to the image diagonal in pixels.
The horizontal place is reliable; the height is not (the mark tends to land low, near the feet).
--offsets: first video frame of each BVH (default 0).
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_side_by_side as bss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("bvh", nargs="+")
    ap.add_argument("--times", default="1,4,8,12,16,20,24,28,32,36")
    ap.add_argument("--focal", type=float)
    ap.add_argument("--offsets")
    ap.add_argument("--tile-width", type=int, default=360)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = a.focal or float(np.hypot(W, H))
    offs = [int(x) for x in a.offsets.split(",")] if a.offsets else [0] * len(a.bvh)

    tracks = []
    for path in a.bvh:
        names, parent, o, chans, data, ft = bss.parse_bvh(path)
        hi = next(i for i, n in enumerate(names) if n.lower() == "head")
        heads = [bss.fk(names, parent, o, chans, r)[hi] for r in data]
        tracks.append((os.path.basename(path), np.array(heads), ft))

    colors = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0),
              (0, 128, 255), (128, 0, 255)]
    tiles = []
    for t in [float(x) for x in a.times.split(",")]:
        fi = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            continue
        for k, (name, heads, ft) in enumerate(tracks):
            bi = int(round((fi - offs[k]) / fps / ft))
            if not 0 <= bi < len(heads):
                continue
            x, y, z = heads[bi]
            if z >= -1:
                continue
            u, v = W / 2 + f * x / -z, H / 2 - f * y / -z
            c = colors[k % len(colors)]
            cv2.circle(img, (int(u), int(v)), 14, c, 4)
            cv2.putText(img, str(k), (int(u) - 14, int(v) - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.6, c, 5)
        cv2.putText(img, f"{t:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        s = a.tile_width / W
        tiles.append(cv2.resize(img, (a.tile_width, int(H * s))))
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for k, (name, heads, ft) in enumerate(tracks):
        print(k, name, len(heads), "frames")


if __name__ == "__main__":
    main()
