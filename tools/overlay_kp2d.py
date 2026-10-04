#!/usr/bin/env python3
"""Draw 2D keypoints CSV (frame,skeleton_id,bbox_x1..y2, <name>_u,<name>_v ...) on a video frame.

Usage:
  overlay_kp2d.py VIDEO KP2D.csv --frame N -o out.jpg
  overlay_kp2d.py VIDEO KP2D.csv --frame N -o out.jpg --csv2 OTHER.csv --label1 CPU --label2 CoreML

Draws bbox, body bones, all points. Second CSV is drawn in another colour with a legend.
Output: JPEG quality 90, downscaled to max width 1280.
"""
import argparse, csv
import cv2

BONES = [("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
         ("neck", "nose"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
         ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"),
         ("right_knee", "right_ankle"), ("left_ankle", "left_heel"), ("left_ankle", "left_big_toe_tip"),
         ("right_ankle", "right_heel"), ("right_ankle", "right_big_toe_tip")]


def load(path, frame):
    """Return (bbox, {name: (u, v)}) for the first row with this frame, or None."""
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["frame"]) == frame:
                bbox = [float(row[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
                pts = {k[:-2]: (float(row[k]), float(row[k[:-2] + "_v"]))
                       for k in row if k.endswith("_u")}
                return bbox, pts
    return None


def draw(img, data, color):
    bbox, pts = data
    cv2.rectangle(img, tuple(int(v) for v in bbox[:2]), tuple(int(v) for v in bbox[2:]), color, 2)
    for a, b in BONES:
        if a in pts and b in pts:
            cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), color, 3)
    for u, v in pts.values():
        cv2.circle(img, (int(u), int(v)), 4, color, -1)
        cv2.circle(img, (int(u), int(v)), 4, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--csv2")
    ap.add_argument("--label1", default="csv1")
    ap.add_argument("--label2", default="csv2")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, img = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {a.frame} from {a.video}")

    sets = [(a.csv, a.label1, (0, 255, 0))]  # BGR: green
    if a.csv2:
        sets.append((a.csv2, a.label2, (0, 0, 255)))  # red
    caption = f"frame {a.frame}"
    for path, label, color in sets:
        data = load(path, a.frame)
        if data is None:
            print(f"warning: frame {a.frame} not in {path}")
            continue
        draw(img, data, color)
    if a.csv2:
        caption += f"  {a.label1} (green) vs {a.label2} (red)"
    else:
        caption += f"  {a.label1}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(img, caption, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    if a.csv2:
        for i, (_, label, color) in enumerate(sets):
            y = 90 + 40 * i
            cv2.circle(img, (25, y - 8), 10, color, -1)
            cv2.putText(img, label, (45, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, round(h * 1280 / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(a.out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
