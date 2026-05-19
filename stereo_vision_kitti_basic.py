"""
Stereo Vision with YOLOv8 - KITTI Dataset Batch Version
Reads calibration from KITTI .txt files and runs stereo depth estimation
over all 50 image pairs, saving results to disk automatically.
Supports both training (with labels) and testing splits.
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
OUTPUT_DIR = Path(r"E:\IBA\BSCS - 6\Introduction to Computer Vision\Project\Depth_Estimation\output\basic")


# ============ KITTI CALIBRATION PARSER ============

def load_kitti_calib(calib_path):
    """
    Parse a KITTI calibration .txt file.

    KITTI stores projection matrices P0–P3 (3×4) and the
    rectification matrix R0_rect (3×3).

    Returns:
        dict with keys: P0, P1, P2, P3, R0_rect, Tr_velo_to_cam
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

      P3[0, 3] = -fx * baseline   (negative tx in mm)
      baseline  = -P3[0,3] / P2[0,0]

    Returns:
        (focal_length_px, baseline_mm)
    """
    P2 = calib['P2']   # left  camera
    P3 = calib['P3']   # right camera

    focal_length = P2[0, 0]                    # fx in pixels
    baseline_m   = -P3[0, 3] / P3[0, 0]       # metres
    baseline_mm  = baseline_m * 1000.0

    return focal_length, baseline_mm


# ============ KITTI LABEL PARSER (ground truth, training only) ============

def load_kitti_labels(label_path):
    """
    Parse a KITTI label .txt file.

    Each row: type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry

    Returns:
        List of dicts with keys: class_name, bbox, location (x,y,z in cam coords)
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
                'class_name': parts[0],
                'bbox':       tuple(map(float, parts[4:8])),   # x1,y1,x2,y2
                'location':   tuple(map(float, parts[11:14])), # x,y,z (metres)
                'distance_gt': float(parts[13]),                # z = depth
            })
    return labels


# ============ STEREO DETECTOR ============

class StereoVisionDetector:
    def __init__(self, model_path="yolov8m.pt", baseline=540.0, focal_length=721.5):
        self.model        = YOLO(model_path)
        self.baseline     = baseline      # mm
        self.focal_length = focal_length  # px

    def detect_objects(self, image):
        results = self.model(image, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf   = float(box.conf)
            cls_id = int(box.cls)
            detections.append({
                'bbox':       (x1, y1, x2, y2),
                'center':     ((x1 + x2) // 2, (y1 + y2) // 2),
                'conf':       conf,
                'cls_id':     cls_id,
                'class_name': results.names[cls_id],
            })
        return detections

    def get_bbox_center_x(self, bbox):
        x1, _, x2, _ = bbox
        return (x1 + x2) / 2.0

    def calculate_disparity_and_distance(self, left_bbox, right_bbox):
        left_cx  = self.get_bbox_center_x(left_bbox)
        right_cx = self.get_bbox_center_x(right_bbox)
        disparity = left_cx - right_cx

        if disparity <= 0:
            return None

        distance = (self.baseline * self.focal_length) / disparity
        return {'disparity': disparity, 'distance': distance, 'unit': 'mm'}

    def match_bounding_boxes(self, detections_left, detections_right,
                              max_y_diff=5, max_disparity=300, min_disparity=1):
        """
        KITTI images are rectified, so epipolar lines are perfectly horizontal.
        max_y_diff=5 is tight and correct for rectified stereo.
        KITTI baseline ~540mm, so max disparity ~300px covers objects down to ~3m.
        """
        matches    = []
        used_right = set()

        for det_left in detections_left:
            left_center = det_left['center']
            best_match  = None
            best_score  = float('inf')

            for i, det_right in enumerate(detections_right):
                if i in used_right:
                    continue
                if det_left['cls_id'] != det_right['cls_id']:
                    continue

                y_diff = abs(left_center[1] - det_right['center'][1])
                if y_diff > max_y_diff:
                    continue

                disparity = left_center[0] - det_right['center'][0]
                if not (min_disparity <= disparity <= max_disparity):
                    continue

                if y_diff < best_score:
                    best_score = y_diff
                    best_match = (i, det_right)

            if best_match is None:
                continue

            idx, matched_det = best_match
            used_right.add(idx)

            dist_info = self.calculate_disparity_and_distance(
                det_left['bbox'], matched_det['bbox']
            )
            if dist_info is not None:
                matches.append({
                    'left_detection':  det_left,
                    'right_detection': matched_det,
                    'distance_info':   dist_info,
                    'center':          left_center,
                })
        return matches

    def draw_results(self, img_left, img_right, matches,
                     gt_labels=None, output_path=None):
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

        for match in matches:
            left_det  = match['left_detection']
            right_det = match['right_detection']
            distance  = match['distance_info']['distance']

            # Left image — green boxes, distance in metres
            x1, y1, x2, y2 = left_det['bbox']
            cv2.rectangle(img_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{left_det['class_name']} {distance/1000:.2f}m"
            cv2.putText(img_l, label, (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Right image — blue boxes
            x1, y1, x2, y2 = right_det['bbox']
            cv2.rectangle(img_r, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_r, right_det['class_name'], (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        h = max(img_l.shape[0], img_r.shape[0])
        w = img_l.shape[1] + img_r.shape[1]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[:img_l.shape[0], :img_l.shape[1]] = img_l
        result[:img_r.shape[0],  img_l.shape[1]:] = img_r

        if output_path:
            cv2.imwrite(str(output_path), result)

        return result


# ============ BATCH MAIN ============

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialise detector once — model is reused across all 50 pairs
    print("Initialising stereo detector...")
    detector = StereoVisionDetector(model_path="yolov8m.pt")

    total_images = 50
    processed    = 0
    skipped      = 0

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

        # Update detector calibration for this specific pair
        detector.baseline     = baseline
        detector.focal_length = focal_length

        # Load ground-truth labels (training split only) 
        gt_labels = []
        if SPLIT == "training":
            gt_labels = load_kitti_labels(label_path)

        # Detect & match 
        detections_left  = detector.detect_objects(img_left)
        detections_right = detector.detect_objects(img_right)

        matches = detector.match_bounding_boxes(detections_left, detections_right)

        # Console summary 
        print(f"\n[{idx_str}]  fl={focal_length:.1f}px  base={baseline:.1f}mm  "
              f"det_L={len(detections_left)}  det_R={len(detections_right)}  "
              f"matched={len(matches)}")

        for i, match in enumerate(matches, 1):
            det  = match['left_detection']
            dist = match['distance_info']['distance']
            disp = match['distance_info']['disparity']
            print(f"  Object {i}: {det['class_name']:12s}  "
                  f"disp={disp:6.1f}px  dist={dist/1000:.3f}m  "
                  f"conf={det['conf']:.2f}")

        # Save result image 
        out_path = OUTPUT_DIR / f"{idx_str}_basic_result.jpg"
        detector.draw_results(
            img_left, img_right, matches,
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
