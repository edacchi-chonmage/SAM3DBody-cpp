#!/usr/bin/env python
"""Check that a time window of a video is one continuous take with exactly one person."""
import argparse
import sys

import cv2
import numpy as np
import onnxruntime as ort


def ranges(times, step):
    out, i = [], 0
    while i < len(times):
        j = i
        while j + 1 < len(times) and times[j + 1] - times[j] <= step * 1.5:
            j += 1
        out.append(f"{times[i]:.2f}" if i == j else f"{times[i]:.2f}-{times[j]:.2f}")
        i = j + 1
    return ", ".join(out) if out else "-"


def detect(sess, frame, thresh):
    h, w = frame.shape[:2]
    s = 640 / max(h, w)
    nw, nh = round(w * s), round(h * s)
    img = np.full((640, 640, 3), 114, np.uint8)
    img[:nh, :nw] = cv2.resize(frame, (nw, nh))
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255
    out = sess.run(["output0"], {"images": x})[0][0]  # [56, 8400]
    keep = out[4] > thresh
    if not keep.any():
        return []
    b, c = out[:4, keep].T, out[4, keep]
    xy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 2], b[:, 3]], 1)
    idx = np.array(cv2.dnn.NMSBoxes(xy.tolist(), c.tolist(), thresh, 0.45)).flatten()
    res = []
    for i in idx[np.argsort(-c[idx])]:
        x1, y1, bw, bh = xy[i] / s
        res.append((x1, y1, x1 + bw, y1 + bh))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--yolo", default="/Users/masatoshi.koeda/tools/SAM3DBody-cpp/onnx/yolo.onnx")
    ap.add_argument("--jump", type=float, default=0.15)
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    k = max(1, round(vfps / a.fps))
    step = k / vfps
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    sess = ort.InferenceSession(a.yolo, providers=["CPUExecutionProvider"])

    samples = []  # (t, n, box or None)
    n = 0
    while True:
        t0 = a.start + n * step
        ok, frame = cap.read() if n == 0 else (True, None)
        if n > 0:
            for _ in range(k):
                ok, frame = cap.read()
                if not ok:
                    break
        if not ok or t0 > a.end:
            break
        boxes = detect(sess, frame, a.thresh)
        samples.append((t0, len(boxes), boxes[0] if boxes else None))
        n += 1

    print(f"video={a.video} window={a.start}-{a.end}s size={int(W)}x{int(H)} samples={len(samples)}")
    t0s = [s[0] for s in samples if s[1] == 0]
    t2s = [s[0] for s in samples if s[1] >= 2]
    print(f"0 persons: {len(t0s)}  at {ranges(t0s, step)}")
    print(f">=2 persons: {len(t2s)}  at {ranges(t2s, step)}")

    jumps = []
    for p, q in zip(samples, samples[1:]):
        if p[2] is None or q[2] is None:
            continue
        cp = np.array([(p[2][0] + p[2][2]) / 2, (p[2][1] + p[2][3]) / 2])
        cq = np.array([(q[2][0] + q[2][2]) / 2, (q[2][1] + q[2][3]) / 2])
        disp = np.linalg.norm(cq - cp) / W
        hp, hq = p[2][3] - p[2][1], q[2][3] - q[2][1]
        ratio = max(hp, hq) / max(1e-6, min(hp, hq))
        if disp > a.jump or ratio > 1.6:
            jumps.append(q[0])
            print(f"JUMP at {q[0]:.2f}s: displacement={disp:.3f} of width, height ratio={ratio:.2f}")
    if not jumps:
        print("no jump events")

    withp = [s for s in samples if s[2] is not None]
    cut = 0
    if withp:
        hs = [(s[2][3] - s[2][1]) / H for s in withp]
        cut = sum(s[2][1] <= 2 or s[2][3] >= H - 2 for s in withp) / len(withp)
        print(f"median box height: {np.median(hs):.2f} of frame height")
        print(f"box touches top/bottom border: {cut:.1%} of samples")

    N = max(1, len(samples))
    why = []
    if jumps:
        why.append(f"{len(jumps)} jump event(s)")
    if len(t0s) / N > 0.05:
        why.append(f"{len(t0s) / N:.1%} samples without a person")
    if len(t2s) / N > 0.02:
        why.append(f"{len(t2s) / N:.1%} samples with >=2 persons")
    print("VERDICT: CONTINUOUS" if not why else "VERDICT: SUSPECT (" + "; ".join(why) + ")")


if __name__ == "__main__":
    sys.exit(main())
