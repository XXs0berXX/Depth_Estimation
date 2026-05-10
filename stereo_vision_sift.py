"""
Stereo Vision with YOLOv8 and SIFT - Enhanced Version
Uses SIFT feature matching for more robust depth estimation.
Combines YOLOv8 detection with SIFT-based feature tracking.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# ============ STEREO CAMERA CALIBRATION PARAMETERS ============
# iPhone 15 Pro Max Dual Camera Parameters
# Main camera:      48MP, f/1.78, 24mm equivalent
# Ultra-wide:       12MP, f/2.2,  13mm equivalent
# 3x Telephoto:     12MP, f/2.8,  77mm equivalent
# 5x Telephoto:     12MP, f/4.2, 120mm equivalent (Pro Max only)
#
# Baseline: ~9mm between main and ultra-wide optical centres
# (from hardware teardown measurements)

# ── Intrinsic matrix for iPhone 15 Pro Max main camera ──────────────────────
# Calibrated for PORTRAIT mode: 3024 × 4032 px output
#
# Derivation:
#   Sensor:          1/1.28"  →  diagonal ≈ 12.5 mm  →  10.0 mm × 7.5 mm (4:3)
#   Physical f:      ≈ 6.86 mm  (24 mm FF-equiv × 12.5/43.27 crop)
#   Pixel pitch:     7.5 mm / 3024 px = 0.00248 mm/px  (same in both axes)
#   fx = fy:         6.86 / 0.00248 ≈ 2766 px
#   cx:              3024 / 2 = 1512
#   cy:              4032 / 2 = 2016
#
# NOTE: Apple applies per-frame lens-correction to HEIF/JPEG output, so
# residual distortion is very small.  Use near-zero D for processed images.
# For ProRAW / DNG frames, run a proper calibration (e.g. with a checkerboard).
K = np.array([
    [2766,    0, 1512],   # fx,  0, cx
    [   0, 2766, 2016],   #  0, fy, cy
    [   0,    0,    1]
], dtype=np.float64)

# Distortion coefficients  [k1, k2, p1, p2, k3]
# Near-zero because Apple software-corrects lens distortion before saving JPEG/HEIF.
D = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Baseline distance between cameras (main ↔ ultra-wide, in mm)
BASELINE = 160  # mm

# Focal length in pixels — matches fx / fy above
FOCAL_LENGTH = 2766.0  # px


class StereoVisionSIFT:
    def __init__(self, model_path="yolov8n.pt", baseline=BASELINE, focal_length=FOCAL_LENGTH):
        """
        Initialize stereo vision detector with SIFT.

        Args:
            model_path:    Path to YOLOv8 model weights.
            baseline:      Distance between the two cameras in mm.
            focal_length:  Focal length in pixels (fx / fy from K matrix).
        """
        self.model        = YOLO(model_path)
        self.baseline     = baseline
        self.focal_length = focal_length

        # SIFT detector
        self.sift = cv2.SIFT_create()

        # FLANN matcher (fast approximate NN for float descriptors)
        index_params  = dict(algorithm=1, trees=5)   # 1 = FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.matcher  = cv2.FlannBasedMatcher(index_params, search_params)

    # ─────────────────────────────────────────────────────────────────────────
    # Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_objects(self, image):
        """
        Detect objects using YOLOv8.

        Returns:
            List of dicts: {bbox, center, conf, cls_id, class_name}
        """
        results = self.model(image, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf    = float(box.conf)
            cls_id  = int(box.cls)
            detections.append({
                'bbox':       (x1, y1, x2, y2),
                'center':     ((x1 + x2) // 2, (y1 + y2) // 2),
                'conf':       conf,
                'cls_id':     cls_id,
                'class_name': results.names[cls_id],
            })

        return detections

    # ─────────────────────────────────────────────────────────────────────────
    # SIFT feature extraction & matching
    # ─────────────────────────────────────────────────────────────────────────

    def extract_bbox_features(self, image, bbox):
        """
        Extract SIFT keypoints/descriptors from a bounding-box ROI.

        Keypoint coordinates are converted back to full-image space.

        Returns:
            (keypoints, descriptors) — keypoints is [] and descriptors is None
            when the ROI is empty or contains no features.
        """
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1);  x2 = min(image.shape[1], x2)
        y1 = max(0, y1);  y2 = min(image.shape[0], y2)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return [], None

        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.sift.detectAndCompute(roi, None)

        # Translate keypoint coords to full-image space
        adjusted = [
            cv2.KeyPoint(kp.pt[0] + x1, kp.pt[1] + y1, kp.size)
            for kp in keypoints
        ]

        return adjusted, descriptors

    def match_features(self, desc1, desc2):
        """
        Match two SIFT descriptor sets with Lowe's ratio test (threshold 0.75).

        Returns:
            List of good cv2.DMatch objects (may be empty).
        """
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        try:
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good = [
                m for pair in raw_matches
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
            ]
            return good
        except Exception as e:
            print(f"[WARN] FLANN matching failed: {e}")
            return []

    def calculate_disparity_from_features(self, img_left, img_right, bbox_left, bbox_right):
        """
        Compute median horizontal disparity via SIFT matches inside the ROIs.

        Returns:
            Median disparity (float) or None when matching fails / too few inliers.
        """
        kp_l, desc_l = self.extract_bbox_features(img_left,  bbox_left)
        kp_r, desc_r = self.extract_bbox_features(img_right, bbox_right)

        if len(kp_l) < 4 or len(kp_r) < 4:
            return None

        matches = self.match_features(desc_l, desc_r)
        if len(matches) < 3:
            return None

        disparities = [
            kp_l[m.queryIdx].pt[0] - kp_r[m.trainIdx].pt[0]
            for m in matches
            if kp_l[m.queryIdx].pt[0] - kp_r[m.trainIdx].pt[0] > 0
        ]

        if not disparities:
            return None

        return float(np.median(disparities))

    # ─────────────────────────────────────────────────────────────────────────
    # Distance calculation
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_distance(self, disparity):
        """
        Depth from disparity:  Z = (baseline × focal_length) / disparity

        Returns:
            Distance in mm, or None for invalid disparity.
        """
        if disparity is None or disparity <= 0:
            return None
        return (self.baseline * self.focal_length) / disparity

    # ─────────────────────────────────────────────────────────────────────────
    # Stereo matching pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def match_and_measure_stereo(self, img_left, img_right,
                                 detections_left, detections_right,
                                 max_y_diff=50):
        """
        Match YOLOv8 detections between the two views and estimate depth
        using SIFT (falling back to centre-based disparity if SIFT fails).

        Args:
            img_left / img_right:          Source images.
            detections_left / right:       YOLOv8 detection dicts.
            max_y_diff:                    Max y-offset for epipolar matching.

        Returns:
            List of measurement dicts:
            {left_detection, right_detection, disparity, distance, confidence}
        """
        measurements = []

        for det_left in detections_left:
            left_center = det_left['center']
            best_match  = None
            best_score  = float('inf')

            for det_right in detections_right:
                if det_left['cls_id'] != det_right['cls_id']:
                    continue
                y_diff = abs(left_center[1] - det_right['center'][1])
                if y_diff > max_y_diff:
                    continue
                if y_diff < best_score:
                    best_score = y_diff
                    best_match = det_right

            if best_match is None:
                continue

            # Try SIFT-based disparity first; fall back to centre difference
            disparity = self.calculate_disparity_from_features(
                img_left, img_right, det_left['bbox'], best_match['bbox']
            )
            if disparity is None:
                disparity = float(left_center[0] - best_match['center'][0])

            if disparity <= 0:
                continue

            distance = self.calculate_distance(disparity)
            measurements.append({
                'left_detection':  det_left,
                'right_detection': best_match,
                'disparity':       disparity,
                'distance':        distance,
                'confidence':      det_left['conf'],
            })

        return measurements

    # ─────────────────────────────────────────────────────────────────────────
    # Visualisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def draw_bboxes_on_image(self, image, detections, color=(0, 255, 0)):
        """Return a copy of *image* annotated with all detection bounding boxes."""
        img_copy = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['conf']:.2f}"
            cv2.putText(img_copy, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img_copy

    def filter_detections_by_roi(self, detections, roi):
        """
        Keep only detections whose centre lies inside *roi*.

        Args:
            detections: List of detection dicts.
            roi:        (x1, y1, x2, y2) or None (returns all detections).
        """
        if roi is None:
            return detections
        rx1, ry1, rx2, ry2 = roi
        return [
            d for d in detections
            if rx1 <= d['center'][0] <= rx2 and ry1 <= d['center'][1] <= ry2
        ]

    def draw_results(self, img_left, img_right, measurements,
                     roi_left=None, roi_right=None, output_path=None):
        """
        Annotate both images with detections, distances, and optional ROI boxes,
        then return them combined side-by-side.
        """
        img_l = img_left.copy()
        img_r = img_right.copy()

        # Draw ROI rectangles
        for img, roi, tag in [(img_l, roi_left, "ROI - LEFT"),
                               (img_r, roi_right, "ROI - RIGHT")]:
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
                cv2.putText(img, tag, (rx1, max(ry1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        for meas in measurements:
            left_det  = meas['left_detection']
            right_det = meas['right_detection']
            distance  = meas['distance']

            # Left image
            x1, y1, x2, y2 = left_det['bbox']
            cv2.rectangle(img_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{left_det['class_name']} {distance:.1f}mm"
            cv2.putText(img_l, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Right image
            x1, y1, x2, y2 = right_det['bbox']
            cv2.rectangle(img_r, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_r, right_det['class_name'], (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Combine side-by-side (pad height if images differ)
        h = max(img_l.shape[0], img_r.shape[0])
        w = img_l.shape[1] + img_r.shape[1]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[:img_l.shape[0], :img_l.shape[1]]  = img_l
        result[:img_r.shape[0],  img_l.shape[1]:] = img_r

        if output_path:
            cv2.imwrite(str(output_path), result)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Interactive ROI selector
# ─────────────────────────────────────────────────────────────────────────────

def select_roi_interactive(image, title="Select ROI"):
    """
    Let the user draw a rectangular ROI on *image* with the mouse.

    Controls:
        Drag left button  — draw rectangle
        c                 — confirm selection
        r                 — reset and redraw
        q / ESC           — skip ROI (use all detections)

    Returns:
        (x1, y1, x2, y2) or None if the user skipped.
    """
    img_copy    = image.copy()
    roi         = None
    drawing     = False
    start_point = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi, drawing, start_point, img_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing     = True
            start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_copy = image.copy()
            cv2.rectangle(img_copy, start_point, (x, y), (0, 255, 255), 2)
            cv2.imshow(title, img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if start_point:
                x1, y1 = start_point
                roi = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
                img_copy = image.copy()
                cv2.rectangle(img_copy, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
                cv2.putText(img_copy, "ROI selected — press C to confirm, R to reset",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(title, img_copy)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse_callback)
    cv2.imshow(title, img_copy)

    print(f"\n{title}")
    print("  Drag to draw  |  C = confirm  |  R = reset  |  Q / ESC = skip")

    while True:
        # BUG FIX: use waitKey(1) instead of waitKey(0) so the event loop keeps
        # running and the live rectangle preview redraws during mouse drag.
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if roi is not None:
                cv2.destroyWindow(title)
                return roi
            print("  No ROI drawn yet — drag a rectangle first.")

        elif key == ord('r'):
            roi      = None
            img_copy = image.copy()
            cv2.imshow(title, img_copy)
            print("  ROI reset.")

        elif key in (ord('q'), 27):   # q or ESC
            cv2.destroyWindow(title)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ============ CONFIGURE YOUR IMAGE PATHS HERE ============
    left_image_path  = r"E:\IBA\BSCS - 6\Introduction to Computer Vision\Project\Depth_Estimation\dataset\2_left_top.jpeg"
    right_image_path = r"E:\IBA\BSCS - 6\Introduction to Computer Vision\Project\Depth_Estimation\dataset\2_right_top.jpeg"
    output_dir = Path("output/stereo_sift")
    output_dir.mkdir(parents=True, exist_ok=True)

    img_left  = cv2.imread(left_image_path)
    img_right = cv2.imread(right_image_path)

    if img_left is None or img_right is None:
        print("Error: Could not load images")
        print(f"  Left:  {left_image_path}")
        print(f"  Right: {right_image_path}")
        return

    print("Loaded stereo image pair")
    print(f"  Left:  {img_left.shape}")
    print(f"  Right: {img_right.shape}")

    print("\nInitialising SIFT-enhanced stereo vision...")
    stereo = StereoVisionSIFT(
        model_path="yolov8n.pt",
        baseline=BASELINE,
        focal_length=FOCAL_LENGTH,
    )

    print("Detecting objects with YOLOv8...")
    detections_left  = stereo.detect_objects(img_left)
    detections_right = stereo.detect_objects(img_right)
    print(f"  Left image:  {len(detections_left)} detections")
    print(f"  Right image: {len(detections_right)} detections")

    print("\n" + "=" * 70)
    print("BOUNDING BOX VISUALISATION & ROI SELECTION")
    print("=" * 70)

    img_l_bboxes = stereo.draw_bboxes_on_image(img_left,  detections_left)
    img_r_bboxes = stereo.draw_bboxes_on_image(img_right, detections_right)

    print("\n--- SELECT ROI FOR LEFT IMAGE ---")
    roi_left  = select_roi_interactive(img_l_bboxes, "LEFT IMAGE - Select ROI")

    print("\n--- SELECT ROI FOR RIGHT IMAGE ---")
    roi_right = select_roi_interactive(img_r_bboxes, "RIGHT IMAGE - Select ROI")

    dets_l = stereo.filter_detections_by_roi(detections_left,  roi_left)
    dets_r = stereo.filter_detections_by_roi(detections_right, roi_right)
    print(f"\nAfter ROI filter — Left: {len(dets_l)}  Right: {len(dets_r)}")

    print("Matching with SIFT and calculating distances...")
    measurements = stereo.match_and_measure_stereo(img_left, img_right, dets_l, dets_r)

    print(f"\n{'='*70}")
    print(f"STEREO VISION + SIFT RESULTS  ({len(measurements)} objects in ROI)")
    print(f"{'='*70}")

    for i, meas in enumerate(measurements, 1):
        det      = meas['left_detection']
        distance = meas['distance']
        disp     = meas['disparity']
        conf     = meas['confidence']
        print(f"\nObject {i}:")
        print(f"  Class:          {det['class_name']}")
        print(f"  Confidence:     {conf:.2f}")
        print(f"  Disparity:      {disp:.2f} px  (SIFT)")
        print(f"  Distance:       {distance:.2f} mm  ({distance / 1000:.3f} m)")
        print(f"  Position (L):   {det['center']}")

    print(f"\n{'='*70}\n")

    result = stereo.draw_results(
        img_left, img_right, measurements,
        roi_left=roi_left, roi_right=roi_right,
        output_path=output_dir / "stereo_sift_result.jpg",
    )
    print(f"Saved result to: {output_dir / 'stereo_sift_result.jpg'}")

    cv2.imshow("Stereo Vision + SIFT Results (press any key to close)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
