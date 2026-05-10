"""
Stereo Vision with YOLOv8 and SIFT - Enhanced Version
Uses SIFT feature matching for more robust depth estimation
Combines YOLOv8 detection with SIFT-based feature tracking
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# ============ STEREO CAMERA CALIBRATION PARAMETERS ============
# iPhone 15 Pro Dual Camera Parameters
# Main camera: 48MP, f/1.78, 26mm equivalent
# Ultra-wide: 12MP, f/2.2, 13mm equivalent
# Baseline: ~12mm between main and ultra-wide sensors

# Camera matrix (intrinsic parameters) for iPhone 15 Pro
# Calibrated for 3024x4032 image resolution
K = np.array([
    [9800, 0, 1512],      # fx, 0, cx (focal length, principal point x)
    [0, 9800, 2016],      # 0, fy, cy (focal length, principal point y)
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients (iPhone has minimal distortion)
D = np.array([0.05, -0.05, 0, 0, 0.02], dtype=np.float32)

# Baseline distance between cameras (main camera to ultra-wide in mm)
BASELINE = 180  # mm

# Focal length (in pixels) - for iPhone 15 Pro main camera
FOCAL_LENGTH = 9800  # pixels


class StereoVisionSIFT:
    def __init__(self, model_path="yolov8n.pt", baseline=BASELINE, focal_length=FOCAL_LENGTH):
        """
        Initialize stereo vision detector with SIFT
        
        Args:
            model_path: Path to YOLOv8 model
            baseline: Distance between cameras (mm)
            focal_length: Focal length (pixels)
        """
        self.model = YOLO(model_path)
        self.baseline = baseline
        self.focal_length = focal_length
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        
        # FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
    def detect_objects(self, image):
        """
        Detect objects using YOLOv8
        """
        results = self.model(image, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls_id = int(box.cls)
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'conf': conf,
                'cls_id': cls_id,
                'class_name': results.names[cls_id]
            })
        
        return detections
    
    def extract_bbox_features(self, image, bbox):
        """
        Extract SIFT features from bounding box region
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            keypoints, descriptors for the region
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bbox
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return [], None
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect SIFT features
        keypoints, descriptors = self.sift.detectAndCompute(roi, None)
        
        # Convert keypoint coordinates to original image space
        adjusted_keypoints = []
        for kp in keypoints:
            kp_adjusted = cv2.KeyPoint(kp.pt[0] + x1, kp.pt[1] + y1, kp.size)
            adjusted_keypoints.append(kp_adjusted)
        
        return adjusted_keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """
        Match SIFT features between two descriptor sets
        Uses Lowe's ratio test for robust matching
        """
        if desc1 is None or desc2 is None:
            return []
        
        if len(desc1) < 2 or len(desc2) < 2:
            return []
        
        try:
            # knnMatch returns k nearest neighbors
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            return good_matches
        except:
            return []
    
    def calculate_disparity_from_features(self, img_left, img_right, bbox_left, bbox_right):
        """
        Calculate disparity using SIFT feature matching within bounding boxes
        
        Args:
            img_left: Left image
            img_right: Right image
            bbox_left: Bounding box in left image
            bbox_right: Bounding box in right image
            
        Returns:
            Disparity value or None if matching fails
        """
        # Extract features from bounding boxes
        kp_left, desc_left = self.extract_bbox_features(img_left, bbox_left)
        kp_right, desc_right = self.extract_bbox_features(img_right, bbox_right)
        
        if len(kp_left) < 4 or len(kp_right) < 4:
            return None
        
        # Match features
        matches = self.match_features(desc_left, desc_right)
        
        if len(matches) < 3:
            return None
        
        # Calculate average horizontal disparity from matched features
        disparities = []
        for m in matches:
            x_left = kp_left[m.queryIdx].pt[0]
            x_right = kp_right[m.trainIdx].pt[0]
            disp = x_left - x_right
            
            if disp > 0:  # Valid disparity
                disparities.append(disp)
        
        if len(disparities) == 0:
            return None
        
        # Return median disparity (robust to outliers)
        return np.median(disparities)
    
    def calculate_distance(self, disparity):
        """
        Calculate distance from disparity
        Distance = (baseline * focal_length) / disparity
        """
        if disparity <= 0:
            return None
        
        distance = (self.baseline * self.focal_length) / disparity
        return distance
    
    def match_and_measure_stereo(self, img_left, img_right, 
                                 detections_left, detections_right, 
                                 max_y_diff=50):
        """
        Match objects between stereo images and measure distances using SIFT
        
        Args:
            img_left: Left image
            img_right: Right image
            detections_left: YOLOv8 detections from left image
            detections_right: YOLOv8 detections from right image
            max_y_diff: Maximum y-coordinate difference for matching
            
        Returns:
            List of matched measurements
        """
        measurements = []
        
        for det_left in detections_left:
            left_center = det_left['center']
            
            # Find best match in right image
            best_match = None
            best_score = float('inf')
            
            for det_right in detections_right:
                right_center = det_right['center']
                
                # Same class check
                if det_left['cls_id'] != det_right['cls_id']:
                    continue
                
                # Y-coordinate similarity
                y_diff = abs(left_center[1] - right_center[1])
                if y_diff > max_y_diff:
                    continue
                
                if y_diff < best_score:
                    best_score = y_diff
                    best_match = det_right
            
            if best_match is None:
                continue
            
            # Use SIFT to calculate more accurate disparity
            disparity = self.calculate_disparity_from_features(
                img_left, img_right,
                det_left['bbox'], best_match['bbox']
            )
            
            if disparity is None:
                # Fallback to center-based disparity
                x_left = left_center[0]
                x_right = best_match['center'][0]
                disparity = x_left - x_right
            
            if disparity > 0:
                distance = self.calculate_distance(disparity)
                
                measurements.append({
                    'left_detection': det_left,
                    'right_detection': best_match,
                    'disparity': disparity,
                    'distance': distance,
                    'confidence': det_left['conf']
                })
        
        return measurements
    
    def draw_bboxes_on_image(self, image, detections, color=(0, 255, 0)):
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detections
            color: Color for bounding boxes (BGR)
            
        Returns:
            Image with drawn bounding boxes
        """
        img_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            text = f"{det['class_name']} {det['conf']:.2f}"
            cv2.putText(img_copy, text, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_copy
    
    def filter_detections_by_roi(self, detections, roi):
        """
        Filter detections that are inside ROI
        
        Args:
            detections: List of detections
            roi: ROI rectangle (x1, y1, x2, y2)
            
        Returns:
            Filtered detections within ROI
        """
        if roi is None:
            return detections
        
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        filtered = []
        
        for det in detections:
            det_x1, det_y1, det_x2, det_y2 = det['bbox']
            det_center_x, det_center_y = det['center']
            
            # Check if detection center is within ROI
            if (roi_x1 <= det_center_x <= roi_x2 and 
                roi_y1 <= det_center_y <= roi_y2):
                filtered.append(det)
        
        return filtered
    
    def draw_results(self, img_left, img_right, measurements, roi_left=None, roi_right=None, output_path=None):
        """
        Draw bounding boxes and distance information
        """
        img_left_copy = img_left.copy()
        img_right_copy = img_right.copy()
        
        # Draw ROI if provided
        if roi_left is not None:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_left
            cv2.rectangle(img_left_copy, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
            cv2.putText(img_left_copy, "ROI - LEFT", (roi_x1, roi_y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if roi_right is not None:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_right
            cv2.rectangle(img_right_copy, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
            cv2.putText(img_right_copy, "ROI - RIGHT", (roi_x1, roi_y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        for meas in measurements:
            left_det = meas['left_detection']
            right_det = meas['right_detection']
            distance = meas['distance']
            
            # Draw on left image
            x1, y1, x2, y2 = left_det['bbox']
            cv2.rectangle(img_left_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            text = f"{left_det['class_name']} {distance:.1f}mm"
            cv2.putText(img_left_copy, text, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw on right image
            x1, y1, x2, y2 = right_det['bbox']
            cv2.rectangle(img_right_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_right_copy, right_det['class_name'], (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Combine images
        h = max(img_left_copy.shape[0], img_right_copy.shape[0])
        w = img_left_copy.shape[1] + img_right_copy.shape[1]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[0:img_left_copy.shape[0], 0:img_left_copy.shape[1]] = img_left_copy
        result[0:img_right_copy.shape[0], img_left_copy.shape[1]:] = img_right_copy
        
        if output_path:
            cv2.imwrite(str(output_path), result)
        
        return result


def select_roi_interactive(image, title="Select ROI - Draw rectangle (press C to confirm, R to reset)"):
    """
    Interactive ROI selection by drawing rectangle on image
    
    Args:
        image: Input image
        title: Window title
        
    Returns:
        ROI coordinates (x1, y1, x2, y2) or None if cancelled
    """
    img_copy = image.copy()
    roi = None
    drawing = False
    start_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi, drawing, start_point, img_copy
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = image.copy()
                cv2.rectangle(img_copy, start_point, (x, y), (0, 255, 255), 2)
                cv2.imshow(title, img_copy)
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if start_point:
                x1, y1 = start_point
                x2, y2 = x, y
                # Ensure coordinates are in correct order
                roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                img_copy = image.copy()
                cv2.rectangle(img_copy, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
                cv2.putText(img_copy, "ROI Selected (Press C to confirm or R to reset)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(title, img_copy)
    
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse_callback)
    cv2.imshow(title, img_copy)
    
    print(f"\n{title}")
    print("Instructions:")
    print("  - Click and drag to draw ROI rectangle")
    print("  - Press 'c' to confirm ROI")
    print("  - Press 'r' to reset ROI")
    print("  - Press 'q' to skip ROI (use all detections)")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('c'):  # Confirm
            if roi is not None:
                cv2.destroyWindow(title)
                return roi
            else:
                print("No ROI selected. Please draw a rectangle first.")
        
        elif key == ord('r'):  # Reset
            roi = None
            img_copy = image.copy()
            cv2.imshow(title, img_copy)
            print("ROI reset. Draw a new rectangle.")
        
        elif key == ord('q'):  # Skip
            cv2.destroyWindow(title)
            return None
        
        elif key == 27:  # ESC
            cv2.destroyWindow(title)
            return None


def main():
    """
    Main function - SIFT-enhanced stereo vision with ROI selection
    """
    # ============ CONFIGURE YOUR IMAGE PATHS HERE ============
    left_image_path = "dataset/2_left_top.jpeg"      # Change to your left image
    right_image_path = "dataset/2_right_top.jpeg"    # Change to your right image
    output_dir = Path("output/stereo_sift")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images
    img_left = cv2.imread(left_image_path)
    img_right = cv2.imread(right_image_path)
    
    if img_left is None or img_right is None:
        print(f"Error: Could not load images")
        print(f"  Left: {left_image_path}")
        print(f"  Right: {right_image_path}")
        return
    
    print(f"Loaded stereo image pair")
    print(f"  Left: {img_left.shape}")
    print(f"  Right: {img_right.shape}")
    
    # Initialize stereo vision with SIFT
    print("\nInitializing SIFT-enhanced stereo vision...")
    stereo = StereoVisionSIFT(
        model_path="yolov8n.pt",
        baseline=BASELINE,
        focal_length=FOCAL_LENGTH
    )
    
    # Detect objects
    print("Detecting objects with YOLOv8...")
    detections_left = stereo.detect_objects(img_left)
    detections_right = stereo.detect_objects(img_right)
    
    print(f"  Left image: {len(detections_left)} detections")
    print(f"  Right image: {len(detections_right)} detections")
    
    # Show detections and allow ROI selection
    print("\n" + "="*70)
    print("BOUNDING BOX VISUALIZATION & ROI SELECTION")
    print("="*70)
    
    # Draw all bounding boxes for user to see
    img_left_with_bboxes = stereo.draw_bboxes_on_image(img_left, detections_left)
    img_right_with_bboxes = stereo.draw_bboxes_on_image(img_right, detections_right)
    
    # Select ROI for left image
    print("\n--- SELECT ROI FOR LEFT IMAGE ---")
    roi_left = select_roi_interactive(img_left_with_bboxes, 
                                      "LEFT IMAGE - Select ROI for distance measurements")
    
    # Select ROI for right image
    print("\n--- SELECT ROI FOR RIGHT IMAGE ---")
    roi_right = select_roi_interactive(img_right_with_bboxes,
                                       "RIGHT IMAGE - Select ROI for distance measurements")
    
    # Filter detections by ROI
    print("\nFiltering detections by ROI...")
    detections_left_filtered = stereo.filter_detections_by_roi(detections_left, roi_left)
    detections_right_filtered = stereo.filter_detections_by_roi(detections_right, roi_right)
    
    print(f"  Left image - After ROI filter: {len(detections_left_filtered)} detections")
    print(f"  Right image - After ROI filter: {len(detections_right_filtered)} detections")
    
    # Match and measure with SIFT
    print("\nMatching with SIFT features and calculating distances...")
    measurements = stereo.match_and_measure_stereo(
        img_left, img_right,
        detections_left_filtered, detections_right_filtered
    )
    
    print(f"\n{'='*70}")
    print(f"STEREO VISION + SIFT RESULTS ({len(measurements)} objects detected in ROI)")
    print(f"{'='*70}")
    
    for i, meas in enumerate(measurements, 1):
        det = meas['left_detection']
        distance = meas['distance']
        disparity = meas['disparity']
        conf = meas['confidence']
        
        print(f"\nObject {i}:")
        print(f"  Class: {det['class_name']}")
        print(f"  Confidence: {conf:.2f}")
        print(f"  Disparity (SIFT): {disparity:.2f} pixels")
        print(f"  Distance: {distance:.2f} mm ({distance/1000:.2f} m)")
        print(f"  Position (Left image): {det['center']}")
    
    print(f"\n{'='*70}\n")
    
    # Draw and save results
    print("Drawing results...")
    result = stereo.draw_results(
        img_left, img_right, measurements,
        roi_left=roi_left, roi_right=roi_right,
        output_path=output_dir / "stereo_sift_result.jpg"
    )
    
    print(f"Saved result to: {output_dir / 'stereo_sift_result.jpg'}")
    
    # Display
    cv2.imshow("Stereo Vision + SIFT Results with ROI (Press any key to close)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
