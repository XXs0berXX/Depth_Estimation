"""
Stereo Vision with YOLOv8 - Basic Version
Uses camera calibration to calculate distance of bounding boxes
Plug in your own left and right stereo images
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


class StereoVisionDetector:
    def __init__(self, model_path="yolov8m.pt", baseline=BASELINE, focal_length=FOCAL_LENGTH):
        """
        Initialize stereo vision detector.

        Args:
            model_path:    Path to YOLOv8 model weights.
            baseline:      Distance between the two cameras in mm.
            focal_length:  Focal length in pixels (fx / fy from K matrix).
        """
        self.model = YOLO(model_path)
        self.baseline = baseline
        self.focal_length = focal_length
        self.detections_left = None
        self.detections_right = None

    def detect_objects(self, image):
        """
        Detect objects in *image* using YOLOv8.

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

    def get_bbox_center_x(self, bbox):
        """Return the x-coordinate of the bounding-box centre."""
        x1, _, x2, _ = bbox
        return (x1 + x2) / 2.0

    def calculate_disparity_and_distance(self, left_bbox, right_bbox):
        """
        Compute disparity and depth from a matched bounding-box pair.

        Args:
            left_bbox:  (x1, y1, x2, y2) in the left image.
            right_bbox: (x1, y1, x2, y2) in the right image.

        Returns:
            Dict {disparity, distance, unit} or None if disparity is invalid.
        """
        left_cx  = self.get_bbox_center_x(left_bbox)
        right_cx = self.get_bbox_center_x(right_bbox)

        disparity = left_cx - right_cx

        if disparity <= 0:
            return None  # invalid / behind the camera

        distance = (self.baseline * self.focal_length) / disparity

        return {
            'disparity': disparity,
            'distance':  distance,
            'unit':      'mm',
        }

    def match_bounding_boxes(self, detections_left, detections_right, max_y_diff=50):
        """
        Greedily match detections between left and right images.

        Matching criteria:
          - Same class ID
          - Centre y-coordinates within *max_y_diff* pixels (epipolar constraint)

        Args:
            detections_left:  Detections from the left image.
            detections_right: Detections from the right image.
            max_y_diff:       Maximum allowed vertical offset between centres.

        Returns:
            List of match dicts: {left_detection, right_detection, distance_info, center}
        """
        matches = []
        used_right = set()  # prevent double-matching

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
                if disparity < 5 or disparity > 200:  # tighter range for 10-20m objects
                    continue

                # Also require the right detection to be spatially close in x
                x_diff = abs(left_center[0] - det_right['center'][0])
                if x_diff > 200:  # same car shouldn't shift more than 200px
                    continue

                if y_diff < best_score:
                    best_score = y_diff
                    best_match = (i, det_right)

            if best_match is None:
                continue

            idx, matched_det = best_match
            used_right.add(idx)  # mark as used

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

    def draw_results(self, img_left, img_right, matches, output_path=None):
        """
        Annotate both images with bounding boxes and distance labels,
        then combine them side-by-side.

        Args:
            img_left:    Left camera image.
            img_right:   Right camera image.
            matches:     Output of match_bounding_boxes().
            output_path: Optional file path to save the combined image.

        Returns:
            Combined image (numpy array).
        """
        img_l = img_left.copy()
        img_r = img_right.copy()

        for match in matches:
            left_det  = match['left_detection']
            right_det = match['right_detection']
            dist_info = match['distance_info']
            distance  = dist_info['distance']

            # ── Left image ────────────────────────────────────────────────────
            x1, y1, x2, y2 = left_det['bbox']
            cv2.rectangle(img_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{left_det['class_name']} {distance:.1f}mm"
            cv2.putText(img_l, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # ── Right image ───────────────────────────────────────────────────
            x1, y1, x2, y2 = right_det['bbox']
            cv2.rectangle(img_r, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_r, right_det['class_name'], (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        # Combine side-by-side (pad height if images differ)
        h = max(img_l.shape[0], img_r.shape[0])
        w = img_l.shape[1] + img_r.shape[1]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[:img_l.shape[0], :img_l.shape[1]]          = img_l
        result[:img_r.shape[0],  img_l.shape[1]:]         = img_r

        if output_path:
            cv2.imwrite(str(output_path), result)

        return result


def main():
    # ============ CONFIGURE YOUR IMAGE PATHS HERE ============
    left_image_path  = r"E:\IBA\BSCS - 6\Introduction to Computer Vision\Project\Depth_Estimation\dataset\2_left_bottom.jpeg"
    right_image_path = r"E:\IBA\BSCS - 6\Introduction to Computer Vision\Project\Depth_Estimation\dataset\2_right_bottom.jpeg"
    output_dir = Path("output/stereo")
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

    detector = StereoVisionDetector(
        model_path="yolov8m.pt",
        baseline=BASELINE,
        focal_length=FOCAL_LENGTH,
    )

    print("\nDetecting objects...")
    detections_left  = detector.detect_objects(img_left)
    detections_right = detector.detect_objects(img_right)
    print(f"  Left image:  {len(detections_left)} detections")
    print(f"  Right image: {len(detections_right)} detections")

    print("\nMatching detections and calculating distances...")
    matches = detector.match_bounding_boxes(detections_left, detections_right)

    print(f"\n{'='*60}")
    print(f"STEREO VISION RESULTS  ({len(matches)} matched objects)")
    print(f"{'='*60}")

    for i, match in enumerate(matches, 1):
        det  = match['left_detection']
        dist = match['distance_info']['distance']
        disp = match['distance_info']['disparity']
        print(f"\nObject {i}:")
        print(f"  Class:      {det['class_name']}")
        print(f"  Confidence: {det['conf']:.2f}")
        print(f"  Disparity:  {disp:.2f} px")
        print(f"  Distance:   {dist:.2f} mm  ({dist / 1000:.3f} m)")
        print(f"  Position:   {det['center']}")

    print(f"\n{'='*60}\n")

    result = detector.draw_results(
        img_left, img_right, matches,
        output_path=output_dir / "stereo_result.jpg",
    )
    print(f"Saved result to: {output_dir / 'stereo_result.jpg'}")

    cv2.imshow("Stereo Vision Results (press any key to close)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()