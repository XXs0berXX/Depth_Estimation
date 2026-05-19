"""
ensemble_yolo_sift.py
=====================
Ensemble depth-estimation pipeline with two selectable techniques:

  1. SIMPLE  — matches YOLO detections across all 6 stereo pairs and
               uses the bounding-box centre coordinates for disparity
               (identical logic to ensemble_yolo.py).

  2. SIFT    — same 4-camera / 6-pair ensemble structure, but replaces
               the centre-based disparity with SIFT feature matching
               inside each matched ROI pair.  Falls back to centre-based
               disparity when SIFT cannot find enough inliers.

Usage:
    python ensemble_yolo_sift.py          # prompts for technique choice
    python ensemble_yolo_sift.py simple
    python ensemble_yolo_sift.py sift
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import json
from datetime import datetime

# ---------------------------------------------------------------------------
# Camera / lens parameters  (iPhone 15 Pro Max, 4-camera rig)
# ---------------------------------------------------------------------------

BASELINE_H_M    = 0.180                                          # left ↔ right,  18 cm
BASELINE_V_M    = 0.350                                          # top  ↔ bottom, 35 cm
BASELINE_DIAG_M = np.sqrt(BASELINE_H_M**2 + BASELINE_V_M**2)   # diagonal ~39.4 cm

FOCAL_LENGTH_MM_EQ = 28.0
SENSOR_WIDTH_MM_EQ = 36.0

# Image paths (relative to this file's directory)
IMG_TL = 'dataset/3_left_top.jpeg'
IMG_TR = 'dataset/3_right_top.jpeg'
IMG_BL = 'dataset/3_left_bottom.jpeg'
IMG_BR = 'dataset/3_right_bottom.jpeg'

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_and_detect(image_path, model):
    """Load an image and run YOLO inference on it."""
    if not os.path.exists(image_path):
        return None, []
    img = cv2.imread(image_path)
    if img is None:
        return None, []
    detections = []
    for r in model(img, verbose=False):
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            detections.append({
                'box':    (int(x1), int(y1), int(x2), int(y2)),
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                'conf':   float(box.conf[0].cpu().numpy()),
                'class':  cls,
                'label':  model.names[cls] if hasattr(model, 'names') else str(cls),
            })
    return img, detections


def select_roi_for_image(img, window_name):
    """Interactive ROI selection using cv2.selectROI."""
    if img is None:
        return None
    disp = img.copy()
    cv2.putText(disp, f"[{window_name}] Draw ROI → SPACE/ENTER  |  C to skip",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    roi = cv2.selectROI(window_name, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    return roi if roi != (0, 0, 0, 0) else None


def filter_by_roi(detections, roi):
    """Keep only detections whose centre lies inside the ROI."""
    if roi is None:
        return []
    rx, ry, rw, rh = roi
    return [d for d in detections
            if rx <= d['center'][0] <= rx + rw and ry <= d['center'][1] <= ry + rh]


def find_match(ref, candidates, shift_type):
    """Axis-aligned match: x_left (right camera) or y_up (bottom camera)."""
    best, best_score = None, float('inf')
    rcx, rcy = ref['center']
    for c in candidates:
        if c['class'] != ref['class']:
            continue
        cx, cy = c['center']
        if shift_type == 'x_left':
            if cx >= rcx: continue
            score = abs(rcy - cy) * 2.0 + abs(rcx - cx)
        else:  # y_up
            if cy >= rcy: continue
            score = abs(rcx - cx) * 2.0 + abs(rcy - cy)
        if score < best_score:
            best_score, best = score, c
    return best


def find_match_diagonal(ref, candidates, expect_dx_positive, expect_dy_positive):
    """
    Diagonal match — both x AND y must shift in the expected direction.
    expect_dx_positive=True  → candidate cx < ref cx  (target shifted left)
    expect_dy_positive=True  → candidate cy < ref cy  (target shifted up)
    """
    best, best_score = None, float('inf')
    rcx, rcy = ref['center']
    for c in candidates:
        if c['class'] != ref['class']:
            continue
        cx, cy = c['center']
        if expect_dx_positive and cx >= rcx: continue
        if not expect_dx_positive and cx <= rcx: continue
        if expect_dy_positive and cy >= rcy: continue
        if not expect_dy_positive and cy <= rcy: continue
        score = abs(rcx - cx) + abs(rcy - cy)
        if score < best_score:
            best_score, best = score, c
    return best


def draw_det(img, det, depth, color):
    """Draw a labelled bounding box with the estimated depth."""
    if img is None or det is None:
        return
    x1, y1, x2, y2 = det['box']
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{det['label']} {depth:.2f}m",
                (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def build_grid(out_tl, out_tr, out_bl, out_br):
    """Tile the four output images into a 2×2 grid and return it."""
    def resize_for_grid(img, w=800):
        if img is None:
            return np.zeros((100, w, 3), dtype=np.uint8)
        h, ow = img.shape[:2]
        return cv2.resize(img, (w, int(h * w / ow)))

    W = 800
    panels = [('TL (Ref)', out_tl), ('TR', out_tr), ('BL', out_bl), ('BR', out_br)]
    resized = []
    for label, out in panels:
        r = resize_for_grid(out, W)
        cv2.putText(r, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        resized.append(r)

    grid = np.vstack([np.hstack(resized[:2]), np.hstack(resized[2:])])
    if grid.shape[0] > 1000:
        s = 1000 / grid.shape[0]
        grid = cv2.resize(grid, (int(grid.shape[1] * s), 1000))
    return grid


def show_grid(out_tl, out_tr, out_bl, out_br, title="Ensemble Depth Result"):
    """Build and display the 2×2 grid."""
    grid = build_grid(out_tl, out_tr, out_bl, out_br)
    cv2.imshow(title, grid)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_outputs(out, log_entries, technique, script_dir):
    """
    Save annotated images and JSON log to ensemble_output/<timestamp>_<technique>/.

    Files written
    -------------
    tl.jpg, tr.jpg, bl.jpg, br.jpg  — individual annotated camera frames
    grid.jpg                         — 2×2 composite
    log.json                         — structured depth log
    """
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(script_dir, 'ensemble_output', f"{ts}_{technique}")
    os.makedirs(run_dir, exist_ok=True)

    # Individual frames
    for key, fname in [('tl', 'tl.jpg'), ('tr', 'tr.jpg'),
                       ('bl', 'bl.jpg'), ('br', 'br.jpg')]:
        img = out.get(key)
        if img is not None:
            path = os.path.join(run_dir, fname)
            cv2.imwrite(path, img)
            print(f"  Saved {fname}  →  {path}")

    # 2×2 grid
    grid = build_grid(out.get('tl'), out.get('tr'), out.get('bl'), out.get('br'))
    grid_path = os.path.join(run_dir, 'grid.jpg')
    cv2.imwrite(grid_path, grid)
    print(f"  Saved grid.jpg  →  {grid_path}")

    # JSON log — custom encoder converts numpy scalars to native Python types
    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    payload = {
        'timestamp': ts,
        'technique': technique,
        'detections': log_entries,
    }
    json_path = os.path.join(run_dir, 'log.json')
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2, cls=_NpEncoder)
    print(f"  Saved log.json  →  {json_path}")

# ---------------------------------------------------------------------------
# Technique 1 — Simple Ensemble  (same as ensemble_yolo.py)
# ---------------------------------------------------------------------------

def run_simple_ensemble(imgs, dets, f_px):
    """
    Estimate depth using centre-based disparity across all 6 stereo pairs.

    Parameters
    ----------
    imgs : dict  — {'tl': img, 'tr': img, 'bl': img, 'br': img}
    dets : dict  — {'tl': [...], 'tr': [...], 'bl': [...], 'br': [...]}
    f_px : float — focal length in pixels

    Returns
    -------
    out_imgs   : dict — annotated copies of the four images
    log_entries: list — one dict per detected object with depth info
    """
    print("\n[SIMPLE ENSEMBLE] Running centre-based disparity estimation...")

    out = {k: (v.copy() if v is not None else None) for k, v in imgs.items()}
    log_entries = []

    # Draw ROI rectangles
    for key in ('tl', 'tr', 'bl', 'br'):
        roi = dets[key + '_roi']
        if out[key] is not None and roi is not None:
            rx, ry, rw, rh = roi
            cv2.rectangle(out[key], (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    for det_tl in dets['tl']:

        # Axis-aligned matches
        match_tr    = find_match(det_tl,   dets['tr'], 'x_left') if imgs['tr'] is not None else None
        match_bl    = find_match(det_tl,   dets['bl'], 'y_up')   if imgs['bl'] is not None else None
        match_br_h  = find_match(match_bl, dets['br'], 'x_left') if match_bl and imgs['br'] is not None else None
        match_br_v  = find_match(match_tr, dets['br'], 'y_up')   if match_tr and imgs['br'] is not None else None

        # Diagonal matches
        match_br_diag = find_match_diagonal(
            det_tl, dets['br'],
            expect_dx_positive=True, expect_dy_positive=True
        ) if imgs['br'] is not None else None

        match_bl_diag = find_match_diagonal(
            match_tr, dets['bl'],
            expect_dx_positive=False, expect_dy_positive=True
        ) if match_tr and imgs['bl'] is not None else None

        z_estimates = []

        def try_add(name, numerator, disparity):
            if disparity > 0:
                z_estimates.append((name, (f_px * numerator) / disparity))

        if match_tr:
            try_add('TL↔TR', BASELINE_H_M, det_tl['center'][0] - match_tr['center'][0])
        if match_bl and match_br_h:
            try_add('BL↔BR', BASELINE_H_M, match_bl['center'][0] - match_br_h['center'][0])
        if match_bl:
            try_add('TL↔BL', BASELINE_V_M, det_tl['center'][1] - match_bl['center'][1])
        if match_tr and match_br_v:
            try_add('TR↔BR', BASELINE_V_M, match_tr['center'][1] - match_br_v['center'][1])
        if match_br_diag:
            dx = det_tl['center'][0] - match_br_diag['center'][0]
            dy = det_tl['center'][1] - match_br_diag['center'][1]
            try_add('TL↔BR-x', BASELINE_H_M, dx)
            try_add('TL↔BR-y', BASELINE_V_M, dy)
        if match_bl_diag:
            dx = match_bl_diag['center'][0] - match_tr['center'][0]
            dy = match_tr['center'][1]       - match_bl_diag['center'][1]
            try_add('TR↔BL-x', BASELINE_H_M, dx)
            try_add('TR↔BL-y', BASELINE_V_M, dy)

        if not z_estimates:
            print(f"  No valid stereo pairs for '{det_tl['label']}' — skipping.")
            continue

        depths    = [z for _, z in z_estimates]
        avg_depth = float(np.mean(depths))
        std_depth = float(np.std(depths))
        pair_info = '  '.join(f"{n}={z:.3f}m" for n, z in z_estimates)
        print(f"  {det_tl['label']:15s}  avg={avg_depth:.3f}m  "
              f"std={std_depth:.3f}m  ({len(z_estimates)} pairs)  [{pair_info}]")

        log_entries.append({
            'label':     det_tl['label'],
            'conf':      round(det_tl['conf'], 4),
            'center_tl': [round(det_tl['center'][0], 1), round(det_tl['center'][1], 1)],
            'avg_depth_m':  round(avg_depth, 4),
            'std_depth_m':  round(std_depth, 4),
            'num_pairs':    len(z_estimates),
            'pairs':        {n: round(z, 4) for n, z in z_estimates},
        })

        draw_det(out['tl'], det_tl,      avg_depth, (0,   255, 255))
        if match_tr:   draw_det(out['tr'], match_tr,      avg_depth, (255,   0,   0))
        if match_bl:   draw_det(out['bl'], match_bl,      avg_depth, (0,     0, 255))
        br_match = match_br_v or match_br_h or match_br_diag
        if br_match:   draw_det(out['br'], br_match,      avg_depth, (255,   0, 255))

    return out, log_entries

# ---------------------------------------------------------------------------
# Technique 2 — SIFT Ensemble
# ---------------------------------------------------------------------------

class SIFTDisparity:
    """
    Wraps SIFT + FLANN to compute feature-based disparity inside two ROI crops.

    The horizontal disparity is derived from the median x-shift of good SIFT
    matches (Lowe's ratio 0.75).  Falls back to ``None`` when the ROI is too
    small or there are too few inliers.
    """

    def __init__(self):
        self.sift = cv2.SIFT_create()
        index_params  = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.flann    = cv2.FlannBasedMatcher(index_params, search_params)

    def _extract(self, img_bgr, box):
        """Return (keypoints, descriptors) in full-image pixel coordinates."""
        x1, y1, x2, y2 = box
        x1 = max(0, x1); x2 = min(img_bgr.shape[1], x2)
        y1 = max(0, y1); y2 = min(img_bgr.shape[0], y2)
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return [], None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        kps, desc = self.sift.detectAndCompute(gray, None)
        # shift keypoints back to full-image coords
        kps = [cv2.KeyPoint(kp.pt[0] + x1, kp.pt[1] + y1, kp.size) for kp in kps]
        return kps, desc

    def disparity(self, img_ref, box_ref, img_cmp, box_cmp, axis='x'):
        """
        Compute median disparity along *axis* ('x' or 'y') between two ROIs.

        Returns float disparity or None.
        """
        kp_r, desc_r = self._extract(img_ref, box_ref)
        kp_c, desc_c = self._extract(img_cmp, box_cmp)

        if len(kp_r) < 4 or len(kp_c) < 4 or desc_r is None or desc_c is None:
            return None
        if len(desc_r) < 2 or len(desc_c) < 2:
            return None

        try:
            raw = self.flann.knnMatch(desc_r, desc_c, k=2)
        except Exception as e:
            print(f"  [SIFT WARN] FLANN failed: {e}")
            return None

        good = [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]

        if len(good) < 3:
            return None

        if axis == 'x':
            shifts = [kp_r[m.queryIdx].pt[0] - kp_c[m.trainIdx].pt[0] for m in good]
        else:
            shifts = [kp_r[m.queryIdx].pt[1] - kp_c[m.trainIdx].pt[1] for m in good]

        positive = [s for s in shifts if s > 0]
        return float(np.median(positive)) if positive else None


def run_sift_ensemble(imgs, dets, f_px):
    """
    Estimate depth across all 6 stereo pairs using SIFT feature matching
    inside each matched bounding-box pair.  Falls back to centre-based
    disparity when SIFT cannot find enough matches.

    Parameters
    ----------
    imgs : dict  — {'tl': img, 'tr': img, 'bl': img, 'br': img}
    dets : dict  — {'tl': [...], 'tr': [...], 'bl': [...], 'br': [...]}
    f_px : float — focal length in pixels

    Returns
    -------
    out_imgs   : dict — annotated copies of the four images
    log_entries: list — one dict per detected object with depth info
    """
    print("\n[SIFT ENSEMBLE] Running SIFT-based disparity estimation...")

    sift_disp = SIFTDisparity()
    out = {k: (v.copy() if v is not None else None) for k, v in imgs.items()}
    log_entries = []

    # Draw ROI rectangles
    for key in ('tl', 'tr', 'bl', 'br'):
        roi = dets[key + '_roi']
        if out[key] is not None and roi is not None:
            rx, ry, rw, rh = roi
            cv2.rectangle(out[key], (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    def centre_disp_x(a, b):
        return a['center'][0] - b['center'][0]

    def centre_disp_y(a, b):
        return a['center'][1] - b['center'][1]

    pair_status = {}   # tracks per-pair SIFT/fallback for the JSON log

    def sift_or_centre(img_a, det_a, img_b, det_b, axis='x', pair_name='?'):
        """Use SIFT disparity; fall back to centre disparity with verbose logging."""
        d = sift_disp.disparity(img_a, det_a['box'], img_b, det_b['box'], axis=axis)
        if d is not None:
            print(f"    [SIFT OK ] {pair_name}  axis={axis}  disp={d:.2f}px")
            pair_status[pair_name] = 'sift'
            return d
        raw = (det_a['center'][0] - det_b['center'][0] if axis == 'x'
               else det_a['center'][1] - det_b['center'][1])
        if raw > 0:
            print(f"    [FALLBACK] {pair_name}  axis={axis}  disp={raw:.2f}px (SIFT failed — using centre)")
            pair_status[pair_name] = 'fallback'
            return raw
        print(f"    [SKIP    ] {pair_name}  axis={axis}  disp≤0, skipping")
        pair_status[pair_name] = 'skipped'
        return None

    for det_tl in dets['tl']:
        pair_status = {}   # reset for each object

        # Axis-aligned matches (same logic as simple ensemble)
        match_tr    = find_match(det_tl,   dets['tr'], 'x_left') if imgs['tr'] is not None else None
        match_bl    = find_match(det_tl,   dets['bl'], 'y_up')   if imgs['bl'] is not None else None
        match_br_h  = find_match(match_bl, dets['br'], 'x_left') if match_bl and imgs['br'] is not None else None
        match_br_v  = find_match(match_tr, dets['br'], 'y_up')   if match_tr and imgs['br'] is not None else None

        match_br_diag = find_match_diagonal(
            det_tl, dets['br'],
            expect_dx_positive=True, expect_dy_positive=True
        ) if imgs['br'] is not None else None

        match_bl_diag = find_match_diagonal(
            match_tr, dets['bl'],
            expect_dx_positive=False, expect_dy_positive=True
        ) if match_tr and imgs['bl'] is not None else None

        z_estimates = []

        def try_add_sift(name, baseline, img_a, det_a, img_b, det_b, axis='x'):
            if det_a is None or det_b is None:
                return
            d = sift_or_centre(img_a, det_a, img_b, det_b, axis=axis, pair_name=name)
            if d and d > 0:
                z_estimates.append((name, (f_px * baseline) / d))

        # Pair 1 — TL ↔ TR  (horizontal, 18 cm)
        if match_tr:
            try_add_sift('TL↔TR(SIFT)', BASELINE_H_M,
                         imgs['tl'], det_tl, imgs['tr'], match_tr, axis='x')

        # Pair 2 — BL ↔ BR  (horizontal, 18 cm)
        if match_bl and match_br_h:
            try_add_sift('BL↔BR(SIFT)', BASELINE_H_M,
                         imgs['bl'], match_bl, imgs['br'], match_br_h, axis='x')

        # Pair 3 — TL ↔ BL  (vertical, 35 cm)
        if match_bl:
            try_add_sift('TL↔BL(SIFT)', BASELINE_V_M,
                         imgs['tl'], det_tl, imgs['bl'], match_bl, axis='y')

        # Pair 4 — TR ↔ BR  (vertical, 35 cm)
        if match_tr and match_br_v:
            try_add_sift('TR↔BR(SIFT)', BASELINE_V_M,
                         imgs['tr'], match_tr, imgs['br'], match_br_v, axis='y')

        # Pair 5 — TL ↔ BR diagonal
        if match_br_diag:
            try_add_sift('TL↔BR-x(SIFT)', BASELINE_H_M,
                         imgs['tl'], det_tl, imgs['br'], match_br_diag, axis='x')
            try_add_sift('TL↔BR-y(SIFT)', BASELINE_V_M,
                         imgs['tl'], det_tl, imgs['br'], match_br_diag, axis='y')

        # Pair 6 — TR ↔ BL diagonal
        if match_bl_diag and match_tr:
            # BL is right+down relative to TR → x-disparity is reversed
            d_x = sift_or_centre(imgs['bl'], match_bl_diag, imgs['tr'], match_tr, axis='x', pair_name='TR↔BL-x(SIFT)')
            d_y = sift_or_centre(imgs['tr'], match_tr,      imgs['bl'], match_bl_diag, axis='y', pair_name='TR↔BL-y(SIFT)')
            if d_x and d_x > 0:
                z_estimates.append(('TR↔BL-x(SIFT)', (f_px * BASELINE_H_M) / d_x))
            if d_y and d_y > 0:
                z_estimates.append(('TR↔BL-y(SIFT)', (f_px * BASELINE_V_M) / d_y))

        if not z_estimates:
            print(f"  No valid stereo pairs for '{det_tl['label']}' — skipping.")
            continue

        depths    = [z for _, z in z_estimates]
        avg_depth = float(np.mean(depths))
        std_depth = float(np.std(depths))
        pair_info = '  '.join(f"{n}={z:.3f}m" for n, z in z_estimates)
        print(f"  {det_tl['label']:15s}  avg={avg_depth:.3f}m  "
              f"std={std_depth:.3f}m  ({len(z_estimates)} pairs)  [{pair_info}]")

        log_entries.append({
            'label':     det_tl['label'],
            'conf':      round(det_tl['conf'], 4),
            'center_tl': [round(det_tl['center'][0], 1), round(det_tl['center'][1], 1)],
            'avg_depth_m':  round(avg_depth, 4),
            'std_depth_m':  round(std_depth, 4),
            'num_pairs':    len(z_estimates),
            'pairs':        {n: round(z, 4) for n, z in z_estimates},
            'sift_status':  pair_status,
        })

        draw_det(out['tl'], det_tl,      avg_depth, (0,   255, 255))
        if match_tr:   draw_det(out['tr'], match_tr,      avg_depth, (255,   0,   0))
        if match_bl:   draw_det(out['bl'], match_bl,      avg_depth, (0,     0, 255))
        br_match = match_br_v or match_br_h or match_br_diag
        if br_match:   draw_det(out['br'], br_match,      avg_depth, (255,   0, 255))

    return out, log_entries

# ---------------------------------------------------------------------------
# Shared setup — model loading, detection, ROI selection
# ---------------------------------------------------------------------------

def setup(script_dir):
    """Load model, run inference on all 4 images, and collect ROI selections."""
    model_path = os.path.join(script_dir, '..', 'yolov8n.pt')

    print("Loading YOLO model...")
    model = YOLO(model_path)

    print("Running inference on all 4 images...")
    img_tl, raw_tl = load_and_detect(os.path.join(script_dir, IMG_TL), model)
    img_tr, raw_tr = load_and_detect(os.path.join(script_dir, IMG_TR), model)
    img_bl, raw_bl = load_and_detect(os.path.join(script_dir, IMG_BL), model)
    img_br, raw_br = load_and_detect(os.path.join(script_dir, IMG_BR), model)

    if img_tl is None:
        print("Could not load TL image.")
        return None, None, None

    img_h, img_w = img_tl.shape[:2]
    f_px = (img_w * FOCAL_LENGTH_MM_EQ) / SENSOR_WIDTH_MM_EQ

    print("Draw a bounding box on each image. SPACE/ENTER to confirm, C to skip.")
    roi_tl = select_roi_for_image(img_tl, "TL - Draw ROI")
    roi_tr = select_roi_for_image(img_tr, "TR - Draw ROI")
    roi_bl = select_roi_for_image(img_bl, "BL - Draw ROI")
    roi_br = select_roi_for_image(img_br, "BR - Draw ROI")

    imgs = {'tl': img_tl, 'tr': img_tr, 'bl': img_bl, 'br': img_br}

    dets = {
        'tl':     filter_by_roi(raw_tl, roi_tl),
        'tr':     filter_by_roi(raw_tr, roi_tr),
        'bl':     filter_by_roi(raw_bl, roi_bl),
        'br':     filter_by_roi(raw_br, roi_br),
        # stash the ROIs so the rendering functions can draw them
        'tl_roi': roi_tl,
        'tr_roi': roi_tr,
        'bl_roi': roi_bl,
        'br_roi': roi_br,
    }

    print(f"ROI detections — TL:{len(dets['tl'])}  TR:{len(dets['tr'])}  "
          f"BL:{len(dets['bl'])}  BR:{len(dets['br'])}")

    return imgs, dets, f_px

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine technique from command-line arg or interactive prompt
    technique = None
    if len(sys.argv) > 1:
        technique = sys.argv[1].lower()

    if technique not in ('simple', 'sift'):
        print("\nSelect depth estimation technique:")
        print("  1. simple  — centre-based disparity (fast, same as ensemble_yolo.py)")
        print("  2. sift    — SIFT feature matching inside each ROI pair (more robust)")
        choice = input("Enter '1'/'simple' or '2'/'sift' [default: simple]: ").strip().lower()
        # accept numeric shorthand
        if choice == '1':
            choice = 'simple'
        elif choice == '2':
            choice = 'sift'
        technique = choice if choice in ('simple', 'sift') else 'simple'

    print(f"\nUsing technique: {technique.upper()}")

    imgs, dets, f_px = setup(script_dir)
    if imgs is None:
        return

    if technique == 'simple':
        out, log_entries = run_simple_ensemble(imgs, dets, f_px)
        title = "Simple Ensemble Depth Result"
    else:
        out, log_entries = run_sift_ensemble(imgs, dets, f_px)
        title = "SIFT Ensemble Depth Result"

    print("\nSaving outputs...")
    save_outputs(out, log_entries, technique, script_dir)

    show_grid(out['tl'], out['tr'], out['bl'], out['br'], title=title)


if __name__ == "__main__":
    main()
