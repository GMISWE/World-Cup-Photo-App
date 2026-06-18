"""Replicate the app's multi-pass BlazeFace detection in Python and sweep the
test fixtures. Lets us iterate on grid/overlap/gate params without the browser.

Mirrors detectFacesMultiPass in index.html:
  full-frame pass + overlapping grid (cols x rows), each tile upscaled so its
  short side >= 512, boxes mapped back to full-image coords, IoU>0.4 NMS
  (keep larger area), then confidence gate (strong>=0.6; filter only if
  >=gateN strong boxes exist).

Usage: .venv/bin/python scripts/detect_sweep.py [cols rows overlap gateN]
"""
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "test-fixtures"
MODEL = "/tmp/blaze_short.tflite"

NAMES = ["p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08",
         "p09", "p10", "p11", "p12", "x03", "x07", "x11"]
# ground truth from independent GMI gpt-4o vision count (x03/x11 adjudicated by eye)
TRUTH = {"p01": 1, "p02": 2, "p03": 3, "p04": 4, "p05": 5, "p06": 6, "p07": 7,
         "p08": 8, "p09": 9, "p10": 10, "p11": 11, "p12": 12,
         "x03": 3, "x07": 7, "x11": 7}

_det = None
def detector():
    global _det
    if _det is None:
        opts = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.2,
        )
        _det = mp_vision.FaceDetector.create_from_options(opts)
    return _det


def detect_rgb(rgb):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = detector().detect(img)
    out = []
    for d in res.detections:
        b = d.bounding_box
        sc = d.categories[0].score if d.categories else 1.0
        out.append([float(b.origin_x), float(b.origin_y), float(b.width), float(b.height), float(sc)])
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return 0 if u <= 0 else inter / u


def contained(a, b):
    """fraction of smaller box a that lies inside b"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = a[2] * a[3]
    return 0 if aa <= 0 else inter / aa


def nms(boxes, t=0.4):
    # keep higher-SCORE boxes; drop a later box if it overlaps a kept one (IoU)
    # OR is mostly contained in it (handles full-frame vs tile size mismatch).
    s = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    for b in s:
        if not any(iou(b, k) > t or contained(b, k) > 0.6 for k in keep):
            keep.append(b)
    return keep


def multipass(rgb, cols, rows, overlap, conf, nms_iou=0.4, min_short=512):
    H, W = rgb.shape[:2]
    allb = list(detect_rgb(rgb))
    bw, bh = W / cols, H / rows
    px, py = bw * overlap, bh * overlap
    for r in range(rows):
        for c in range(cols):
            X = max(0, round(c * bw - px)); Y = max(0, round(r * bh - py))
            Wd = min(round(bw + 2 * px), W - X); Hd = min(round(bh + 2 * py), H - Y)
            if Wd <= 0 or Hd <= 0:
                continue
            sc = max(1.0, min_short / min(Wd, Hd))
            crop = rgb[Y:Y + Hd, X:X + Wd]
            if sc > 1.0:
                crop = cv2.resize(crop, (round(Wd * sc), round(Hd * sc)), interpolation=cv2.INTER_LINEAR)
            for b in detect_rgb(crop):
                allb.append([X + b[0] / sc, Y + b[1] / sc, b[2] / sc, b[3] / sc, b[4]])
    kept = [b for b in allb if b[4] >= conf]
    return nms(kept, nms_iou)


def load_rgb(name):
    bgr = cv2.imread(str(FIX / f"{name}.png"))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run(cols, rows, overlap, conf, nms_iou=0.4, min_short=512):
    print(f"\n== grid {cols}x{rows} overlap={overlap} conf>={conf} nms={nms_iou} minShort={min_short} ==")
    print(f"{'img':5} {'truth':>5} {'det':>4} {'diff':>5}")
    rows_out = []
    total_err = 0
    for n in NAMES:
        rgb = load_rgb(n)
        k = multipass(rgb, cols, rows, overlap, conf, nms_iou, min_short)
        d = len(k)
        t = TRUTH[n]
        err = d - t
        total_err += abs(err)
        rows_out.append((n, t, d, err))
        flag = "" if err == 0 else ("  <<" if abs(err) > 1 else "  <")
        print(f"{n:5} {t:>5} {d:>4} {err:>+5}{flag}")
    exact = sum(1 for _, t, d, _ in rows_out if t == d)
    within1 = sum(1 for _, t, d, _ in rows_out if abs(t - d) <= 1)
    print(f"exact={exact}/15  within1={within1}/15  total_abs_err={total_err}")
    return rows_out


if __name__ == "__main__":
    if len(sys.argv) >= 5:
        nms_iou = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
        run(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), nms_iou)
    else:
        # production params (square/portrait branch): 4x3 grid, 25% overlap,
        # conf>=0.5, NMS IoU 0.35 (+ containment merge).
        run(4, 3, 0.25, 0.5, 0.35)
