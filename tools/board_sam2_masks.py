"""Segment a skateboard through a video with SAM 2.1 (PyTorch MPS).

usage: board_sam2_masks.py VIDEO --seed-frame N --box x1,y1,x2,y2
         [--pos x,y;x,y] [--neg x,y;x,y] -o OUT.npz [--preview OUT.mp4] [--scale 0.5]
Prompt coordinates are full-resolution pixels. Run with sam2env/bin/python.
npz: masks uint8 [F,h,w] (at --scale), scale, fps, area (per frame, pixels).
"""
import argparse, os, shutil, subprocess, tempfile, time
import numpy as np, cv2, torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from sam2.build_sam import build_sam2_video_predictor

CKPT = "/Users/masatoshi.koeda/tools/skateboard/sam2env/ckpt/sam2.1_hiera_small.pt"
CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

def pts(s):
    return np.array([[float(v) for v in p.split(",")] for p in s.split(";")], np.float32) if s else np.zeros((0, 2), np.float32)

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--seed-frame", type=int, required=True)
ap.add_argument("--box", required=True); ap.add_argument("--pos", default=""); ap.add_argument("--neg", default="")
ap.add_argument("-o", required=True); ap.add_argument("--preview"); ap.add_argument("--scale", type=float, default=0.5)
a = ap.parse_args()

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("mps available:", torch.backends.mps.is_available(), "device:", dev)
t0 = time.time()
tmp = tempfile.mkdtemp()
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    f = cv2.resize(f, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{tmp}/{len(frames):05d}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95]); frames.append(f)
F, (h, w) = len(frames), frames[0].shape[:2]

pred = build_sam2_video_predictor(CFG, CKPT, device=dev)
st = pred.init_state(tmp, offload_video_to_cpu=True)
p, n = pts(a.pos) * a.scale, pts(a.neg) * a.scale
xy = np.concatenate([p, n]); lab = np.array([1] * len(p) + [0] * len(n), np.int32)
box = np.array([float(v) for v in a.box.split(",")], np.float32) * a.scale
pred.add_new_points_or_box(st, a.seed_frame, 1, points=xy if len(xy) else None, labels=lab if len(xy) else None, box=box)

masks = np.zeros((F, h, w), np.uint8)
for rev in (False, True):
    for i, _, lg in pred.propagate_in_video(st, start_frame_idx=a.seed_frame, reverse=rev):
        masks[i] = (lg[0, 0] > 0).cpu().numpy()
area = masks.reshape(F, -1).sum(1)
np.savez_compressed(a.o, masks=masks, scale=a.scale, fps=fps, area=area)
print(f"runtime {time.time()-t0:.1f}s frames {F} device {dev}")

if a.preview:
    raw = tempfile.mktemp(suffix=".mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, f in enumerate(frames):
        o = f.copy(); m = masks[i] > 0
        o[m] = (0.5 * o[m] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.putText(o, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        vw.write(o)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p", a.preview], check=True)
    os.remove(raw)
shutil.rmtree(tmp)
