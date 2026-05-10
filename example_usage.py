"""
Simple Example: Using Stereo Vision with Your Own Images
This file demonstrates the easiest way to get started
"""

from stereo_vision_basic import StereoVisionDetector
from stereo_vision_sift import StereoVisionSIFT
import cv2
from pathlib import Path

# ============================================================
# EXAMPLE 1: Basic Stereo Vision (Simple & Fast)
# ============================================================

def example_basic():
    """
    Simple example using basic stereo vision
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Stereo Vision")
    print("=" * 60)
    
    # Your image paths - CHANGE THESE
    left_image_path = "stereo_left.jpg"
    right_image_path = "stereo_right.jpg"
    
    # Load images
    img_left = cv2.imread(left_image_path)
    img_right = cv2.imread(right_image_path)
    
    if img_left is None or img_right is None:
        print(f"Error: Could not load images")
        return
    
    # Create detector with your camera parameters
    # IMPORTANT: Update these values based on your camera!
    detector = StereoVisionDetector(
        model_path="yolov8n.pt",
        baseline=120,        # Distance between cameras in mm
        focal_length=800     # Focal length in pixels
    )
    
    # Detect objects in both images
    print("Detecting objects...")
    detections_left = detector.detect_objects(img_left)
    detections_right = detector.detect_objects(img_right)
    
    print(f"Found {len(detections_left)} objects in left image")
    print(f"Found {len(detections_right)} objects in right image")
    
    # Match bounding boxes and calculate distances
    print("Calculating distances...")
    matches = detector.match_bounding_boxes(
        detections_left, 
        detections_right,
        max_y_diff=50  # Adjust based on image rectification
    )
    
    # Print results
    print(f"\nDetected {len(matches)} objects:\n")
    for i, match in enumerate(matches, 1):
        det = match['left_detection']
        dist = match['distance_info']['distance']
        print(f"{i}. {det['class_name']:10s} at {det['center']:20s} | "
              f"Distance: {dist:8.1f} mm ({dist/1000:5.2f} m)")
    
    # Draw results
    result = detector.draw_results(img_left, img_right, matches)
    
    # Save result
    output_path = Path("output/stereo") / "example_basic.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"\nResult saved to: {output_path}")
    
    # Display (optional)
    cv2.imshow("Stereo Vision Results", result)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# EXAMPLE 2: SIFT-Enhanced Stereo Vision (More Accurate)
# ============================================================

def example_sift():
    """
    Example using SIFT-enhanced stereo vision
    Better for challenging conditions
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: SIFT-Enhanced Stereo Vision")
    print("=" * 60)
    
    # Your image paths - CHANGE THESE
    left_image_path = "stereo_left.jpg"
    right_image_path = "stereo_right.jpg"
    
    # Load images
    img_left = cv2.imread(left_image_path)
    img_right = cv2.imread(right_image_path)
    
    if img_left is None or img_right is None:
        print(f"Error: Could not load images")
        return
    
    # Create SIFT-enhanced detector
    stereo = StereoVisionSIFT(
        model_path="yolov8n.pt",
        baseline=120,        # Distance between cameras in mm
        focal_length=800     # Focal length in pixels
    )
    
    # Detect objects
    print("Detecting objects with YOLOv8...")
    detections_left = stereo.detect_objects(img_left)
    detections_right = stereo.detect_objects(img_right)
    
    print(f"Found {len(detections_left)} objects in left image")
    print(f"Found {len(detections_right)} objects in right image")
    
    # Match with SIFT and calculate distances
    print("Matching with SIFT features and calculating distances...")
    measurements = stereo.match_and_measure_stereo(
        img_left, img_right,
        detections_left, detections_right
    )
    
    # Print results
    print(f"\nDetected {len(measurements)} objects:\n")
    for i, meas in enumerate(measurements, 1):
        det = meas['left_detection']
        dist = meas['distance']
        print(f"{i}. {det['class_name']:10s} at {det['center']:20s} | "
              f"Distance: {dist:8.1f} mm ({dist/1000:5.2f} m)")
    
    # Draw results
    result = stereo.draw_results(img_left, img_right, measurements)
    
    # Save result
    output_path = Path("output/stereo_sift") / "example_sift.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"\nResult saved to: {output_path}")
    
    # Display (optional)
    cv2.imshow("SIFT-Enhanced Stereo Vision Results", result)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# EXAMPLE 3: Batch Processing Multiple Stereo Pairs
# ============================================================

def example_batch_processing():
    """
    Process multiple stereo image pairs
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)
    
    # Define your stereo pairs
    stereo_pairs = [
        ("stereo_left_1.jpg", "stereo_right_1.jpg"),
        ("stereo_left_2.jpg", "stereo_right_2.jpg"),
        ("stereo_left_3.jpg", "stereo_right_3.jpg"),
    ]
    
    detector = StereoVisionDetector(
        model_path="yolov8n.pt",
        baseline=120,
        focal_length=800
    )
    
    output_dir = Path("output/stereo_batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pair_idx, (left_path, right_path) in enumerate(stereo_pairs, 1):
        print(f"\nProcessing pair {pair_idx}...")
        
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        
        if img_left is None or img_right is None:
            print(f"  Skipped: Could not load images")
            continue
        
        # Detect and measure
        detections_left = detector.detect_objects(img_left)
        detections_right = detector.detect_objects(img_right)
        matches = detector.match_bounding_boxes(detections_left, detections_right)
        
        print(f"  Found {len(matches)} objects")
        
        # Save result
        result = detector.draw_results(img_left, img_right, matches)
        output_path = output_dir / f"pair_{pair_idx}.jpg"
        cv2.imwrite(str(output_path), result)
        print(f"  Saved to: {output_path}")


# ============================================================
# EXAMPLE 4: Custom Configuration
# ============================================================

def example_custom_config():
    """
    Advanced example with custom parameters
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 60)
    
    left_image_path = "stereo_left.jpg"
    right_image_path = "stereo_right.jpg"
    
    img_left = cv2.imread(left_image_path)
    img_right = cv2.imread(right_image_path)
    
    if img_left is None or img_right is None:
        print("Error: Could not load images")
        return
    
    # Custom camera parameters
    BASELINE = 100        # mm - distance between cameras
    FOCAL_LENGTH = 750    # pixels - focal length of your camera
    
    detector = StereoVisionDetector(
        model_path="yolov8n.pt",
        baseline=BASELINE,
        focal_length=FOCAL_LENGTH
    )
    
    # Detect objects
    detections_left = detector.detect_objects(img_left)
    detections_right = detector.detect_objects(img_right)
    
    # Custom matching parameters
    matches = detector.match_bounding_boxes(
        detections_left,
        detections_right,
        max_y_diff=100  # More lenient y-coordinate tolerance
    )
    
    print(f"Detected {len(matches)} objects")
    
    # Filter results (only keep high-confidence detections)
    high_conf_matches = [m for m in matches if m['left_detection']['conf'] > 0.7]
    
    print(f"High confidence objects: {len(high_conf_matches)}")
    
    for match in high_conf_matches:
        det = match['left_detection']
        dist = match['distance_info']['distance']
        print(f"  {det['class_name']}: {dist/1000:.2f} m")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\nStereo Vision Examples\n")
    
    # Run examples (uncomment to use)
    
    # Basic example
    example_basic()
    
    # SIFT example (more accurate but slower)
    # example_sift()
    
    # Batch processing
    # example_batch_processing()
    
    # Custom configuration
    # example_custom_config()
    
    print("\n✓ Done!")
