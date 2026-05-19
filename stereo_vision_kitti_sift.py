"""
Stereo Vision with YOLOv8 and SIFT - KITTI Batch Version
Reads calibration from KITTI .txt files and runs SIFT-based stereo depth
estimation over all 50 image pairs, saving results to disk automatically.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# ============ KITTI DATASET PATHS ============
KITTI_ROOT = Path(r"E:\IBA\BSCS - 6\Introduction to Computer Vision\Project\kitti-dataset-50\kitti-dataset-50")

SPLIT = "training"   # "training" or "testing"

LEFT_IMG_DIR  = KITTI_ROOT / f"data_object_image_2/{SPLIT}/image_2"
RIGHT_IMG_DIR = KITTI_ROOT / f"data_object_image_3/{SPLIT}/image_3"
CALIB_DIR     = KITTI_ROOT / f"data_object_calib/{SPLIT}/calib"
LABEL_DIR     = KITTI_ROOT / "data_object_label_2/training/label_2"  # training only

# Output folder
OUTPUT_DIR = Path(r"E:\IBA\BSCS - 6\Introduction to Computer Vision\Project\Depth_Estimation\output\sift")


# ============ KITTI CALIBRATION PARSER ============

def load_kitti_calib(calib_path):
    """
    Parse a KITTI calibration .txt file.
    Returns dict with keys: P0, P1, P2, P3, R0_rect, Tr_velo_to_cam
    """
    calib = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, values = line.split(':', 1)
            calib[key.strip()] = np.array(
                [float(v) for v in values.split()]
            )

    result = {}
    for key in ('P0', 'P1', 'P2', 'P3'):
        if key in calib:
            result[key] = calib[key].reshape(3, 4)
    if 'R0_rect' in calib:
        result['R0_rect'] = calib['R0_rect'].reshape(3, 3)
    if 'Tr_velo_to_cam' in calib:
        result['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)

    return result


def get_focal_and_baseline(calib):
    """
    Extract focal length (px) and baseline (mm) from KITTI calibration.

    KITTI convention:
      P2 = left  camera projection matrix
      P3 = right camera projection matrix
      baseline = -P3[0,3] / P3[0,0]   (in metres → convert to mm)

    Returns:
        (focal_length_px, baseline_mm)
    """
    P2 = calib['P2']
    P3 = calib['P3']

    focal_length = P2[0, 0]                  # fx in pixels
    baseline_m   = -P3[0, 3] / P3[0, 0]     # metres
    baseline_mm  = baseline_m * 1000.0

    return focal_length, baseline_mm


# ============ KITTI LABEL PARSER (ground truth, training only) ============

def load_kitti_labels(label_path):
    """
    Parse a KITTI label .txt file.
    Returns list of dicts: {class_name, bbox, location, distance_gt}
    """
    labels = []
    if not Path(label_path).exists():
        return labels

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == 'DontCare':
                continue
            labels.append({
                'class_name':  parts[0],
                'bbox':        tuple(map(float, parts[4:8])),
                'location':    tuple(map(float, parts[11:14])),
                'distance_gt': float(parts[13]),
            })
    return labels


# ============ SIFT STEREO DETECTOR ============

class StereoVisionSIFT:
    def __init__(self, model_path="yolov8n.pt", baseline=540.0, focal_length=721.5):
        """
        Args:
            model_path:    Path to YOLOv8 model weights.
            baseline:      Distance between the two cameras in mm.
            focal_length:  Focal length in pixels (fx from KITTI P2).
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

        adjusted = [
            cv2.KeyPoint(kp.pt[0] + x1, kp.pt[1] + y1, kp.size)
            for kp in keypoints
        ]
        return adjusted, descriptors

    def match_features(self, desc1, desc2):
        """
        Match two SIFT descriptor sets with Lowe's ratio test (threshold 0.75).
        Returns list of good cv2.DMatch objects (may be empty).
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

    def calculate_disparity_from_features(self, img_left, img_right,
                                           bbox_left, bbox_right):
        """
        Compute median horizontal disparity via SIFT matches inside the ROIs.
        Returns median disparity (float) or None when matching fails.
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
        """Z = (baseline × focal_length) / disparity  →  mm"""
        if disparity is None or disparity <= 0:
            return None
        return (self.baseline * self.focal_length) / disparity

    # ─────────────────────────────────────────────────────────────────────────
    # Stereo matching pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def match_and_measure_stereo(self, img_left, img_right,
                                  detections_left, detections_right,
                                  max_y_diff=5):
        """
        Match detections between views and estimate depth using SIFT
        (falling back to centre-based disparity if SIFT fails).

        max_y_diff=5 is tight and correct for rectified KITTI stereo.
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

                disparity_check = left_center[0] - det_right['center'][0]
                if not (1 <= disparity_check <= 300):
                    continue

                if y_diff < best_score:
                    best_score = y_diff
                    best_match = det_right

            if best_match is None:
                continue

            # Try SIFT-based disparity; fall back to centre difference
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
    # Visualisation
    # ─────────────────────────────────────────────────────────────────────────

    def draw_results(self, img_left, img_right, measurements,
                     gt_labels=None, output_path=None):
        """
        Annotate both images with detections and distances, then save
        them combined side-by-side.
        """
        img_l = img_left.copy()
        img_r = img_right.copy()

        # Draw ground-truth boxes in yellow (left image only)
        if gt_labels:
            for lbl in gt_labels:
                x1, y1, x2, y2 = map(int, lbl['bbox'])
                cv2.rectangle(img_l, (x1, y1), (x2, y2), (0, 255, 255), 1)
                gt_label = f"GT {lbl['class_name']} {lbl['distance_gt']:.1f}m"
                cv2.putText(img_l, gt_label, (x1, max(y1 - 12, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        for meas in measurements:
            left_det  = meas['left_detection']
            right_det = meas['right_detection']
            distance  = meas['distance']

            # Left image — green boxes, distance in metres
            x1, y1, x2, y2 = left_det['bbox']
            cv2.rectangle(img_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{left_det['class_name']} {distance/1000:.2f}m"
            cv2.putText(img_l, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Right image — blue boxes
            x1, y1, x2, y2 = right_det['bbox']
            cv2.rectangle(img_r, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_r, right_det['class_name'], (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        h = max(img_l.shape[0], img_r.shape[0])
        w = img_l.shape[1] + img_r.shape[1]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[:img_l.shape[0], :img_l.shape[1]]  = img_l
        result[:img_r.shape[0],  img_l.shape[1]:] = img_r

        if output_path:
            cv2.imwrite(str(output_path), result)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialise detector once — model is reused across all 50 pairs
    print("Initialising SIFT-enhanced stereo vision detector...")
    stereo = StereoVisionSIFT(model_path="yolov8n.pt")

    total_images  = 50
    processed     = 0
    skipped       = 0

    for image_idx in range(total_images):
        idx_str = f"{image_idx:06d}"

        left_path  = LEFT_IMG_DIR  / f"{idx_str}.png"
        right_path = RIGHT_IMG_DIR / f"{idx_str}.png"
        calib_path = CALIB_DIR     / f"{idx_str}.txt"
        label_path = LABEL_DIR     / f"{idx_str}.txt"

        # Load images 
        img_left  = cv2.imread(str(left_path))
        img_right = cv2.imread(str(right_path))

        if img_left is None or img_right is None:
            print(f"[{idx_str}] SKIP — image(s) not found")
            skipped += 1
            continue

        # Load calibration 
        calib        = load_kitti_calib(calib_path)
        focal_length, baseline = get_focal_and_baseline(calib)

        # Update detector calibration for this pair
        stereo.baseline     = baseline
        stereo.focal_length = focal_length

        #  Load ground-truth labels (training split only) 
        gt_labels = []
        if SPLIT == "training":
            gt_labels = load_kitti_labels(label_path)

        # Detect & match 
        detections_left  = stereo.detect_objects(img_left)
        detections_right = stereo.detect_objects(img_right)

        measurements = stereo.match_and_measure_stereo(
            img_left, img_right, detections_left, detections_right
        )

        # Console summary 
        print(f"\n[{idx_str}]  fl={focal_length:.1f}px  base={baseline:.1f}mm  "
              f"det_L={len(detections_left)}  det_R={len(detections_right)}  "
              f"matched={len(measurements)}")

        for i, meas in enumerate(measurements, 1):
            det  = meas['left_detection']
            dist = meas['distance']
            disp = meas['disparity']
            print(f"  Object {i}: {det['class_name']:12s}  "
                  f"disp={disp:6.1f}px  dist={dist/1000:.3f}m  "
                  f"conf={meas['confidence']:.2f}")

        # Save result image 
        out_path = OUTPUT_DIR / f"{idx_str}_sift_result.jpg"
        stereo.draw_results(
            img_left, img_right, measurements,
            gt_labels=gt_labels if SPLIT == "training" else None,
            output_path=out_path,
        )

        processed += 1

    # Final summary 
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE — processed {processed}, skipped {skipped}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
