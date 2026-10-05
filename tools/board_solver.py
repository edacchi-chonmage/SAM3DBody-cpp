#!/usr/bin/env python3
"""Per-frame skateboard pose from BVH feet + SAM2 board masks."""
import sys, csv, time, argparse
import numpy as np, cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot, Slerp
sys.path.insert(0, "/Users/masatoshi.koeda/tools/SAM3DBody-cpp/tools")
import bvh_side_by_side as bss

B = "/Users/masatoshi.koeda/tools/skateboard/"
ap = argparse.ArgumentParser()
ap.add_argument("--iou-thr", type=float, default=0.2)
ap.add_argument("--contact-px", type=float, default=60.0)
ap.add_argument("--axis-deg", type=float, default=15.0)
ap.add_argument("--maxfev", type=int, default=800)
args = ap.parse_args()
T0 = time.time()
names, parent, offs, chans, data, ft = bss.parse_bvh(B + "skate.bvh")
N = len(data)
P = np.array([bss.fk(names, parent, offs, chans, r) for r in data])
J = lambda n: P[:, names.index(n)]
M = np.load(B + "board/cam_M.npy")
Z = np.load(B + "board/masks.npz"); masks = Z["masks"]; SC = float(Z["scale"])
rows = list(csv.reader(open(B + "board/kp2d.csv"))); h = rows[0]
kp = {int(r[0]) - 1: np.array([float(x) for x in r[6:]]).reshape(-1, 2) for r in rows[1:] if r[1] == "0"}
kn = [h[6 + 2 * i][:-2] for i in range(70)]
pairs = [("left_hip","lThigh"),("right_hip","rThigh"),("left_knee","lShin"),("right_knee","rShin"),("left_ankle","lFoot"),("right_ankle","rFoot"),
         ("left_shoulder","lShldr"),("right_shoulder","rShldr"),("left_elbow","lForeArm"),("right_elbow","rForeArm"),("left_wrist","lHand"),("right_wrist","rHand")]

def dlt(X):
    hh = np.c_[X, np.ones(len(X))] @ M.T
    return hh[:, :2] / hh[:, 2:3]

# 1. per-frame 2D similarity correction
S = []; resid = []
for f in range(N):
    A = np.array([0, 0, 1.0]); Mat = np.array([[1, 0, 0], [0, 1, 0.]])
    if f in kp:
        src = dlt(np.array([P[f, names.index(b)] for a, b in pairs]))
        dst = np.array([kp[f][kn.index(a)] for a, b in pairs])
        m, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS)
        if m is not None: Mat = m
        resid.append(np.median(np.linalg.norm(src @ Mat[:, :2].T + Mat[:, 2] - dst, axis=1)))
    S.append(Mat)
def proj(f, X):
    return dlt(X) @ S[f][:, :2].T + S[f][:, 2]
print("similarity-corrected median residual: median over frames %.1f px (p90 %.1f)" % (np.median(resid), np.percentile(resid, 90)))

# 2. cuboid
x0, x1, y0, y1, z0, z1 = -40, 40, -9, 0, -10.5, 10.5
CORN = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
DECK = np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]], float)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]

# 3. stage 1
def unit(v): return v / (np.linalg.norm(v) + 1e-12)
lt, rt = J("toe3-1.L"), J("toe3-1.R"); la, ra = J("lFoot"), J("rFoot")
cL, cR = 0.5 * (la + lt), 0.5 * (ra + rt)
feet_mid = 0.5 * (cL + cR)
# Board normal = ground normal: plane fitted through the foot contact points of frames
# where both feet are level (per-frame foot directions are too noisy for the roll/pitch).
lvl = np.abs(cL[:, 1] - cR[:, 1]) < 10
G = np.r_[cL[lvl], cR[lvl]]; Gc = G.mean(0)
g = np.linalg.svd(G - Gc)[2][2]; g = g if g[1] > 0 else -g
print("ground normal %s (%.1f deg from +Y), %d level frames" % (np.round(g, 3), np.degrees(np.arccos(g[1])), lvl.sum()))
# Long axis = feet-to-feet direction on that plane, smoothed over +-5 frames.
A = np.array([unit(cR[f] - cL[f]) for f in range(N)])
for f in range(1, N):
    if A[f] @ A[f - 1] < 0: A[f] = -A[f]
A = np.array([A[max(0, f - 5):f + 6].mean(0) for f in range(N)])
R1 = []; T1 = []
for f in range(N):
    a = unit(A[f] - (A[f] @ g) * g)
    R1.append(np.c_[a, g, np.cross(a, g)]); T1.append(feet_mid[f] - 7 * g)
R1 = np.array(R1); T1 = np.array(T1)
# Ground plane: the wheels' bottom, 7 cm (deck) + 9 cm (trucks/wheels) under the contact plane.
# The silhouette fixes the board's image position but only weakly its depth/height, so the
# mask fit is kept above this plane and near the feet.
P0 = Gc - 16 * g

# silhouettes in cropped mask space
MH, MW = masks.shape[1:]
crops = []
for f in range(N):
    ys, xs = np.nonzero(masks[f])
    if len(xs) == 0: crops.append(None); continue
    m_ = 40
    cx0, cx1 = max(xs.min() - m_, 0), min(xs.max() + m_, MW); cy0, cy1 = max(ys.min() - m_, 0), min(ys.max() + m_, MH)
    crops.append((cx0, cy0, masks[f, cy0:cy1, cx0:cx1] > 0))
def iou(f, t, R):
    c = crops[f]
    if c is None: return 0.0
    cx0, cy0, mk = c
    pts = proj(f, CORN @ R.T + t) * SC - np.array([cx0, cy0])
    if not np.isfinite(pts).all() or np.abs(pts).max() > 1e5: return 0.0
    hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
    im = np.zeros(mk.shape, np.uint8); cv2.fillConvexPoly(im, hull, 1)
    im = im > 0
    u = (im | mk).sum()
    return (im & mk).sum() / u if u else 0.0

i1 = np.array([iou(f, T1[f], R1[f]) for f in range(N)])

# feet classification: feet over a near-horizontal board strip
def mask_axis(mk):
    ys, xs = np.nonzero(mk)
    if len(xs) < 20: return None
    c = np.cov(np.c_[xs, ys].T); w, v = np.linalg.eigh(c); return np.degrees(np.arctan2(v[1, 1], v[0, 1]))
FT = ["left_heel", "left_big_toe_tip", "right_heel", "right_big_toe_tip"]
touch = np.zeros(N, bool); axok = np.zeros(N, bool); adiff = np.full(N, np.nan); cdist = np.full((N, 2), np.nan)
for f in range(N):
    mk = masks[f] > 0
    if f not in kp or not mk.any(): continue
    dist = cv2.distanceTransform((~mk).astype(np.uint8), cv2.DIST_L2, 5) / SC
    d = []
    for side in ("left", "right"):
        ds = []
        for nm in (side + "_heel", side + "_big_toe_tip"):
            u, v = kp[f][kn.index(nm)] * SC
            ui, vi = int(round(min(max(u, 0), mk.shape[1] - 1))), int(round(min(max(v, 0), mk.shape[0] - 1)))
            ds.append(dist[vi, ui] + np.hypot(u - ui, v - vi) / SC)
        d.append(min(ds))
    # The 2D heel/toe keypoints sit 20-40 px above the deck even when standing on it,
    # so "touch" = both feet over the board's horizontal extent and within --contact-px
    # (default 60) of it, rather than literally on the mask.
    ys, xs = np.nonzero(mk); x0, x1, yc = xs.min() / SC, xs.max() / SC, ys.mean() / SC
    over = all(x0 - 30 <= kp[f][kn.index(nm)][0] <= x1 + 30 and kp[f][kn.index(nm)][1] <= yc for nm in FT)
    cdist[f] = d; touch[f] = over and np.median(d) <= args.contact_px
    # A board flat under the feet is a near-horizontal strip from this low front camera;
    # flips, pops and rail stands tilt it (measured 30-76 deg vs 0-5 deg when riding).
    am = mask_axis(mk)
    if am is not None:
        h = abs(am) % 180; h = min(h, 180 - h); adiff[f] = h; axok[f] = h < args.axis_deg
raw = touch & axok
print("raw: touch %d, flat %d, both %d of %d" % (touch.sum(), axok.sum(), raw.sum(), N))
feet = np.array([np.median(np.pad(raw.astype(int), 2, mode="edge")[f:f + 5]) >= 0.5 for f in range(N)])
def runs_of(b):
    out = []; s_ = 0
    for f in range(1, N + 1):
        if f == N or b[f] != b[s_]: out.append([s_, f - 1]); s_ = f
    return out
while True:
    rr = runs_of(feet); short = [r for r in rr if r[1] - r[0] + 1 < 4]
    if not short or len(rr) == 1: break
    r = min(short, key=lambda r: r[1] - r[0]); feet[r[0]:r[1] + 1] = not feet[r[0]]
print("stage1: %d feet frames, %d mask frames" % (feet.sum(), (~feet).sum()))

# 5. stage 2
def ang(Ra, Rb): return np.degrees(Rot.from_matrix(Ra.T @ Rb).magnitude())
def rx(d): return Rot.from_euler("x", d, degrees=True).as_matrix()
def ry(d): return Rot.from_euler("y", d, degrees=True).as_matrix()
def solve(f, tp, Rp, nxt_stage1=True):
    def cost(p):
        t = tp + p[:3]; R = (Rot.from_rotvec(np.radians(p[3:])) * Rot.from_matrix(Rp0)).as_matrix()
        below = np.clip(-((CORN @ R.T + t - P0) @ g), 0, None).sum()   # cm of box under the ground
        return (1 - iou(f, t, R) + 0.002 * np.linalg.norm(t - tp) + 0.002 * ang(R, Rp)
                + 0.02 * max(0, np.linalg.norm(t - feet_mid[f]) - 45) / 10 + 0.01 * below
                # depth along the camera axis (BVH z) barely changes the silhouette, so tie it to the feet
                + 0.05 * max(0, abs(t[2] - feet_mid[f][2]) - 25) / 10)
    starts = [Rp, Rp @ rx(90), Rp @ rx(180), Rp @ rx(270), Rp @ ry(180), R1[f] @ rx(90)]
    best = None
    for Rs in starts:
        Rp0 = Rs
        r = minimize(cost, np.zeros(6), method="Powell", options=dict(maxfev=args.maxfev, xtol=1e-2, ftol=1e-4))
        if best is None or r.fun < best[0]:
            best = (r.fun, tp + r.x[:3], (Rot.from_rotvec(np.radians(r.x[3:])) * Rot.from_matrix(Rs)).as_matrix())
    t, R = best[1], best[2]
    return t, R, iou(f, t, R)

tF = T1.copy(); RF = R1.copy(); iF = i1.copy()
for f in range(N):
    if feet[f]: continue
    tp, Rp = (tF[f - 1], RF[f - 1]) if f > 0 else (T1[f], R1[f])
    tF[f], RF[f], iF[f] = solve(f, tp, Rp)
tB = T1.copy(); RB = R1.copy(); iB = i1.copy()
for f in range(N - 1, -1, -1):
    if feet[f]: continue
    tp, Rp = (tB[f + 1], RB[f + 1]) if f < N - 1 else (T1[f], R1[f])
    tB[f], RB[f], iB[f] = solve(f, tp, Rp)
tt = tF.copy(); RR = RF.copy(); ii = iF.copy(); src = np.where(feet, "feet", "mask").astype(object)
useB = (~feet) & (iB > iF)
tt[useB] = tB[useB]; RR[useB] = RB[useB]; ii[useB] = iB[useB]
print("backward chosen on %d of %d mask frames" % (useB.sum(), (~feet).sum()))

# 6. joins
isf = feet.copy()
t_out = tt.copy(); R_out = RR.copy()
for b in range(1, N):
    if isf[b] != isf[b - 1]:
        for k, w in zip((5, 4, 3, 2, 1), (1/6, 2/6, 3/6, 4/6, 5/6)):
            g = b - k
            if g < 0: continue
            sl = Slerp([0, 1], Rot.from_matrix(np.stack([RR[g], RR[b]])))
            R_out[g] = sl(w).as_matrix(); t_out[g] = (1 - w) * tt[g] + w * tt[b]
# iou after blending
ii_out = np.array([iou(f, t_out[f], R_out[f]) for f in range(N)])

# 7. outputs
eul = Rot.from_matrix(R_out).as_euler("ZYX", degrees=True)
eul = np.unwrap(eul, period=360, axis=0)
with open(B + "board/board_pose.csv", "w") as fo:
    fo.write("frame,tx,ty,tz,rz,ry,rx,source,iou\n")
    for f in range(N):
        fo.write("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%.4f\n" % (f, *t_out[f], *eul[f], src[f], ii_out[f]))
with open(B + "board/board_2d.csv", "w") as fo:
    fo.write("frame,u1,v1,u2,v2,u3,v3,u4,v4\n")
    for f in range(N):
        fo.write("%d,%s\n" % (f, ",".join("%.2f" % v for v in proj(f, DECK @ R_out[f].T + t_out[f]).ravel())))
# convention check
Rb = np.array([Rot.from_euler("z", e[0], degrees=True).as_matrix() @ Rot.from_euler("y", e[1], degrees=True).as_matrix()
               @ Rot.from_euler("x", e[2], degrees=True).as_matrix() for e in eul])
print("Euler convention max |Rz@Ry@Rx - R| = %.2e" % np.abs(Rb - R_out).max())

def runs(mask):
    out = []; s = None
    for f in range(N + 1):
        v = f < N and mask[f]
        if v and s is None: s = f
        if not v and s is not None: out.append((s, f - 1)); s = None
    return out
srcs = np.array(src, dtype=str)
seg = []; s = 0
for f in range(1, N + 1):
    if f == N or srcs[f] != srcs[s]: seg.append((s, f - 1, srcs[s])); s = f
print("segments (frame range, source):", seg)
for k in ("feet", "mask"):
    m_ = srcs == k
    print("mean IoU %s: %.3f (%d frames)" % (k, ii_out[m_].mean() if m_.any() else float("nan"), m_.sum()))
fm = np.nonzero(srcs == "feet")[0]; offs_ = []
for f in fm:
    ys, xs = np.nonzero(masks[f])
    if len(xs): offs_.append(np.linalg.norm(proj(f, T1[f][None])[0] - np.array([xs.mean(), ys.mean()]) / SC))
print("feet frames: median 2D offset stage1 centre vs mask centroid: %.1f px (n=%d)" % (np.median(offs_) if offs_ else float("nan"), len(offs_)))
print("frames IoU<0.3: %d, ranges: %s" % ((ii_out < 0.3).sum(), runs(ii_out < 0.3)))

# 8. check image
cap = cv2.VideoCapture(B + "skate_pro.mp4"); tiles = []
for t in (0.5, 2.0, 3.0, 3.9, 4.5, 5.0, 6.5, 8.8, 11.2, 13.5, 14.5, 15.5):
    f = int(round(t * 29.97)); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    if not ok: im = np.zeros((1080, 1920, 3), np.uint8)
    mk = cv2.resize(masks[f], (1920, 1080), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, cs, -1, (0, 0, 255), 3)
    pp = proj(f, CORN @ R_out[f].T + t_out[f])
    for i, j in EDGES: cv2.line(im, tuple(int(v) for v in pp[i]), tuple(int(v) for v in pp[j]), (255, 80, 0), 3)
    ys, xs = np.nonzero(mk); c = (int(xs.mean()), int(ys.mean())) if len(xs) else tuple(int(v) for v in pp.mean(0))
    pad = cv2.copyMakeBorder(im, 500, 500, 700, 700, cv2.BORDER_CONSTANT)
    cr = pad[c[1] + 500 - 250:c[1] + 500 + 250, c[0] + 700 - 350:c[0] + 700 + 350]
    cr = cv2.resize(cr, (350, 250), interpolation=cv2.INTER_AREA)
    cv2.putText(cr, "%.1fs f%d %s iou%.2f" % (t, f, srcs[f], ii_out[f]), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(cr)
rowsimg = [np.hstack(tiles[i:i + 3]) for i in range(0, 12, 3)]
cv2.imwrite(B + "board/solver_check.jpg", np.vstack(rowsimg), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("elapsed %.0f s" % (time.time() - T0))
