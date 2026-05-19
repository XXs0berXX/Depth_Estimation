import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- Camera baseline parameters ---
BASELINE_H_M    = 0.180                                          # left ↔ right,  18 cm
BASELINE_V_M    = 0.350                                          # top  ↔ bottom, 35 cm
BASELINE_DIAG_M = np.sqrt(BASELINE_H_M**2 + BASELINE_V_M**2)   # diagonal,      ~39.4 cm

FOCAL_LENGTH_MM_EQ = 28.0
SENSOR_WIDTH_MM_EQ = 36.0

IMG_TL = 'dataset/3_left_top.jpeg'
IMG_TR = 'dataset/3_right_top.jpeg'
IMG_BL = 'dataset/3_left_bottom.jpeg'
IMG_BR = 'dataset/3_right_bottom.jpeg'

# ---------------------------------------------------------------------------

def load_and_detect(image_path, model):
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
    if img is None:
        return None
    disp = img.copy()
    cv2.putText(disp, f"[{window_name}] Draw ROI → SPACE/ENTER  |  C to skip",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    roi = cv2.selectROI(window_name, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    return roi if roi != (0, 0, 0, 0) else None

def filter_by_roi(detections, roi):
    if roi is None:
        return []
    rx, ry, rw, rh = roi
    return [d for d in detections
            if rx <= d['center'][0] <= rx + rw and ry <= d['center'][1] <= ry + rh]

# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------

def draw_det(img, det, depth, color):
    if img is None or det is None:
        return
    x1, y1, x2, y2 = det['box']
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{det['label']} {depth:.2f}m",
                (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'yolov8n.pt')

    print("Loading YOLO model...")
    model = YOLO(model_path)

    print("Running inference on all 4 images...")
    img_tl, dets_tl = load_and_detect(os.path.join(script_dir, IMG_TL), model)
    img_tr, dets_tr = load_and_detect(os.path.join(script_dir, IMG_TR), model)
    img_bl, dets_bl = load_and_detect(os.path.join(script_dir, IMG_BL), model)
    img_br, dets_br = load_and_detect(os.path.join(script_dir, IMG_BR), model)

    if img_tl is None:
        print("Could not load TL image.")
        return

    img_h, img_w = img_tl.shape[:2]
    f_px = (img_w * FOCAL_LENGTH_MM_EQ) / SENSOR_WIDTH_MM_EQ

    print("Draw a bounding box on each image. SPACE/ENTER to confirm, C to skip.")
    roi_tl = select_roi_for_image(img_tl, "TL - Draw ROI")
    roi_tr = select_roi_for_image(img_tr, "TR - Draw ROI")
    roi_bl = select_roi_for_image(img_bl, "BL - Draw ROI")
    roi_br = select_roi_for_image(img_br, "BR - Draw ROI")

    dets_tl = filter_by_roi(dets_tl, roi_tl)
    dets_tr = filter_by_roi(dets_tr, roi_tr)
    dets_bl = filter_by_roi(dets_bl, roi_bl)
    dets_br = filter_by_roi(dets_br, roi_br)

    print(f"ROI detections — TL:{len(dets_tl)}  TR:{len(dets_tr)}  "
          f"BL:{len(dets_bl)}  BR:{len(dets_br)}")

    out_tl = img_tl.copy()
    out_tr = img_tr.copy() if img_tr is not None else None
    out_bl = img_bl.copy() if img_bl is not None else None
    out_br = img_br.copy() if img_br is not None else None

    for out, roi in [(out_tl, roi_tl), (out_tr, roi_tr),
                     (out_bl, roi_bl), (out_br, roi_br)]:
        if out is not None and roi is not None:
            rx, ry, rw, rh = roi
            cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    # ── Depth estimation using all 6 stereo pairs ──────────────────────────────
    for det_tl in dets_tl:

        # ── Axis-aligned matches ───────────────────────────────────────────────
        match_tr = find_match(det_tl,   dets_tr, 'x_left') if img_tr is not None else None
        match_bl = find_match(det_tl,   dets_bl, 'y_up')   if img_bl is not None else None

        # BR via horizontal path (from BL) and vertical path (from TR)
        match_br_h = find_match(match_bl, dets_br, 'x_left') if match_bl and img_br is not None else None
        match_br_v = find_match(match_tr, dets_br, 'y_up')   if match_tr and img_br is not None else None

        # ── Diagonal matches ───────────────────────────────────────────────────
        # TL → BR : BR is right+down, so same object in BR appears left+up vs TL
        match_br_diag = find_match_diagonal(
            det_tl, dets_br,
            expect_dx_positive=True,   # br_cx < tl_cx
            expect_dy_positive=True    # br_cy < tl_cy
        ) if img_br is not None else None

        # TR → BL : BL is left+down, so same object in BL appears right+up vs TR
        match_bl_diag = find_match_diagonal(
            match_tr, dets_bl,
            expect_dx_positive=False,  # bl_cx > tr_cx
            expect_dy_positive=True    # bl_cy < tr_cy
        ) if match_tr and img_bl is not None else None

        # ── Collect depth estimates from every valid pair ──────────────────────
        z_estimates = []

        def try_add(name, numerator, disparity):
            if disparity > 0:
                z_estimates.append((name, (f_px * numerator) / disparity))

        # Pair 1 — TL ↔ TR  (horizontal, 18 cm)
        if match_tr:
            try_add('TL↔TR', BASELINE_H_M,
                    det_tl['center'][0] - match_tr['center'][0])

        # Pair 2 — BL ↔ BR  (horizontal, 18 cm)
        if match_bl and match_br_h:
            try_add('BL↔BR', BASELINE_H_M,
                    match_bl['center'][0] - match_br_h['center'][0])

        # Pair 3 — TL ↔ BL  (vertical, 35 cm)
        if match_bl:
            try_add('TL↔BL', BASELINE_V_M,
                    det_tl['center'][1] - match_bl['center'][1])

        # Pair 4 — TR ↔ BR  (vertical, 35 cm)
        if match_tr and match_br_v:
            try_add('TR↔BR', BASELINE_V_M,
                    match_tr['center'][1] - match_br_v['center'][1])

        # Pair 5 — TL ↔ BR  diagonal: x-component (18 cm) + y-component (35 cm)
        if match_br_diag:
            dx = det_tl['center'][0] - match_br_diag['center'][0]
            dy = det_tl['center'][1] - match_br_diag['center'][1]
            try_add('TL↔BR-x', BASELINE_H_M, dx)
            try_add('TL↔BR-y', BASELINE_V_M, dy)

        # Pair 6 — TR ↔ BL  diagonal: x-component (18 cm) + y-component (35 cm)
        if match_bl_diag:
            dx = match_bl_diag['center'][0] - match_tr['center'][0]   # bl_cx > tr_cx
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

        # Colour-code each camera's annotation
        draw_det(out_tl, det_tl,      avg_depth, (0,   255, 255))   # yellow
        if match_tr:      draw_det(out_tr, match_tr,      avg_depth, (255,   0,   0))  # blue
        if match_bl:      draw_det(out_bl, match_bl,      avg_depth, (0,     0, 255))  # red
        # prefer the vertically-derived BR match; fall back to horizontal or diagonal
        br_match = match_br_v or match_br_h or match_br_diag
        if br_match:      draw_det(out_br, br_match,      avg_depth, (255,   0, 255))  # magenta

    # ── 2×2 result grid ────────────────────────────────────────────────────────
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

    cv2.imshow("Ensemble Depth Result", grid)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()