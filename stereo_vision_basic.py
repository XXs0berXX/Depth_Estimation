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


class StereoVisionDetector:
    def __init__(self, model_path="yolov8n.pt", baseline=BASELINE, focal_length=FOCAL_LENGTH):
        """
        Initialize stereo vision detector
        
        Args:
            model_path: Path to YOLOv8 model
            baseline: Distance between left and right cameras (mm)
            focal_length: Focal length in pixels
        """
        self.model = YOLO(model_path)
        self.baseline = baseline
        self.focal_length = focal_length
        self.detections_left = None
        self.detections_right = None
        
    def detect_objects(self, image):
        """
        Detect objects in image using YOLOv8
        
        Returns:
            List of detections with [x1, y1, x2, y2, confidence, class_id]
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
    
    def get_bbox_center_x(self, bbox):
        """Get x-coordinate of bounding box center"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0
    
    def calculate_disparity_and_distance(self, left_bbox, right_bbox):
        """
        Calculate disparity and distance between matched bounding boxes
        
        Args:
            left_bbox: Bounding box in left image
            right_bbox: Bounding box in right image
            
        Returns:
            Dictionary with disparity and distance info
        """
        left_center_x = self.get_bbox_center_x(left_bbox)
        right_center_x = self.get_bbox_center_x(right_bbox)
        
        # Disparity: difference in x-coordinates
        disparity = left_center_x - right_center_x
        
        if disparity <= 0:
            return None  # Invalid disparity
        
        # Distance = (baseline * focal_length) / disparity
        distance = (self.baseline * self.focal_length) / disparity
        
        return {
            'disparity': disparity,
            'distance': distance,
            'unit': 'mm'  # Based on baseline unit
        }
    
    def match_bounding_boxes(self, detections_left, detections_right, max_y_diff=50):
        """
        Match bounding boxes between left and right images
        Assumes objects are roughly at same y-coordinate
        
        Args:
            detections_left: Detections from left image
            detections_right: Detections from right image
            max_y_diff: Maximum y-coordinate difference for matching
            
        Returns:
            List of matched pairs with distance info
        """
        matches = []
        
        for det_left in detections_left:
            left_center = det_left['center']
            
            # Find matching detection in right image (same class, similar y-coordinate)
            best_match = None
            best_score = float('inf')
            
            for det_right in detections_right:
                right_center = det_right['center']
                
                # Check if same class
                if det_left['cls_id'] != det_right['cls_id']:
                    continue
                
                # Check if y-coordinates are similar
                y_diff = abs(left_center[1] - right_center[1])
                if y_diff > max_y_diff:
                    continue
                
                # Score based on y-difference (lower is better)
                score = y_diff
                
                if score < best_score:
                    best_score = score
                    best_match = det_right
            
            if best_match is not None:
                dist_info = self.calculate_disparity_and_distance(
                    det_left['bbox'], best_match['bbox']
                )
                
                if dist_info is not None:
                    matches.append({
                        'left_detection': det_left,
                        'right_detection': best_match,
                        'distance_info': dist_info,
                        'center': left_center
                    })
        
        return matches
    
    def draw_results(self, img_left, img_right, matches, output_path=None):
        """
        Draw bounding boxes and distance information on images
        
        Args:
            img_left: Left image
            img_right: Right image
            matches: Matched detections with distance info
            output_path: Optional path to save result
        """
        img_left_copy = img_left.copy()
        img_right_copy = img_right.copy()
        
        for match in matches:
            left_det = match['left_detection']
            right_det = match['right_detection']
            dist_info = match['distance_info']
            
            # Draw on left image
            x1, y1, x2, y2 = left_det['bbox']
            cv2.rectangle(img_left_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            distance = dist_info['distance']
            text = f"{left_det['class_name']} {distance:.1f}mm"
            cv2.putText(img_left_copy, text, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw on right image
            x1, y1, x2, y2 = right_det['bbox']
            cv2.rectangle(img_right_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_right_copy, right_det['class_name'], (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Combine images side by side
        h = max(img_left_copy.shape[0], img_right_copy.shape[0])
        w = img_left_copy.shape[1] + img_right_copy.shape[1]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[0:img_left_copy.shape[0], 0:img_left_copy.shape[1]] = img_left_copy
        result[0:img_right_copy.shape[0], img_left_copy.shape[1]:] = img_right_copy
        
        if output_path:
            cv2.imwrite(str(output_path), result)
        
        return result


def main():
    """
    Main function to process stereo images
    """
    # ============ CONFIGURE YOUR IMAGE PATHS HERE ============
    left_image_path = "dataset/2_left_top.jpeg"      # Change to your left image
    right_image_path = "dataset/2_right_top.jpeg"    # Change to your right image
    output_dir = Path("output/stereo")
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
    
    # Initialize detector
    detector = StereoVisionDetector(
        model_path="yolov8n.pt",
        baseline=BASELINE,
        focal_length=FOCAL_LENGTH
    )
    
    # Detect objects in both images
    print("\nDetecting objects...")
    detections_left = detector.detect_objects(img_left)
    detections_right = detector.detect_objects(img_right)
    
    print(f"  Left image: {len(detections_left)} detections")
    print(f"  Right image: {len(detections_right)} detections")
    
    # Match bounding boxes and calculate distances
    print("\nMatching detections and calculating distances...")
    matches = detector.match_bounding_boxes(detections_left, detections_right)
    
    print(f"\n{'='*60}")
    print(f"STEREO VISION RESULTS ({len(matches)} objects detected)")
    print(f"{'='*60}")
    
    for i, match in enumerate(matches, 1):
        det = match['left_detection']
        dist = match['distance_info']['distance']
        disp = match['distance_info']['disparity']
        
        print(f"\nObject {i}:")
        print(f"  Class: {det['class_name']}")
        print(f"  Confidence: {det['conf']:.2f}")
        print(f"  Disparity: {disp:.2f} pixels")
        print(f"  Distance: {dist:.2f} mm ({dist/1000:.2f} m)")
        print(f"  Position: {det['center']}")
    
    print(f"\n{'='*60}\n")
    
    # Draw results
    print("Drawing results...")
    result = detector.draw_results(
        img_left, img_right, matches,
        output_path=output_dir / "stereo_result.jpg"
    )
    
    print(f"Saved result to: {output_dir / 'stereo_result.jpg'}")
    
    # Display results
    cv2.imshow("Stereo Vision Results (Press any key to close)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
