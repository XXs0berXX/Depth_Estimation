"""
Utility Tool: Camera Parameter Finder and Stereo Diagnostics
Helps you determine the right camera parameters for your setup
"""

import cv2
import numpy as np
from pathlib import Path

class StereoDiagnostics:
    """
    Diagnostic tools for stereo vision setup
    """
    
    @staticmethod
    def estimate_focal_length_from_images(image_path, known_distance_mm, known_size_pixels):
        """
        Estimate focal length from known object size
        
        Args:
            image_path: Path to image containing object
            known_distance_mm: Real-world distance to object (mm)
            known_size_pixels: Size of object in image (pixels)
            
        Returns:
            Estimated focal length in pixels
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        # focal_length = (size_in_pixels × distance) / real_world_size
        # Assuming object real-world size is known or can be estimated
        
        print(f"\nEstimating focal length...")
        print(f"  Image size: {img.shape}")
        print(f"  Object size in image: {known_size_pixels} pixels")
        print(f"  Real-world distance: {known_distance_mm} mm")
        
        # This is a simplified calculation
        # For accurate results, calibrate your camera properly
        focal_length = (known_size_pixels * known_distance_mm) / 100  # Rough estimate
        
        print(f"  Estimated focal length: {focal_length:.1f} pixels")
        return focal_length
    
    @staticmethod
    def check_image_alignment(left_image_path, right_image_path):
        """
        Check if stereo images are properly aligned
        
        Args:
            left_image_path: Path to left image
            right_image_path: Path to right image
        """
        img_left = cv2.imread(left_image_path)
        img_right = cv2.imread(right_image_path)
        
        if img_left is None or img_right is None:
            print("Error: Could not load images")
            return
        
        print("\n" + "="*60)
        print("STEREO IMAGE ALIGNMENT CHECK")
        print("="*60)
        
        # Check image shapes
        print(f"\nImage dimensions:")
        print(f"  Left:  {img_left.shape}")
        print(f"  Right: {img_right.shape}")
        
        if img_left.shape != img_right.shape:
            print("  ⚠ WARNING: Images have different dimensions!")
            print("  Both images should have the same size for proper stereo processing")
        else:
            print("  ✓ Images have matching dimensions")
        
        # Convert to grayscale
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        sift = cv2.SIFT_create()
        kp_left, desc_left = sift.detectAndCompute(gray_left, None)
        kp_right, desc_right = sift.detectAndCompute(gray_right, None)
        
        print(f"\nFeature detection:")
        print(f"  Left features:  {len(kp_left)}")
        print(f"  Right features: {len(kp_right)}")
        
        if len(kp_left) < 50 or len(kp_right) < 50:
            print("  ⚠ WARNING: Too few features detected!")
            print("  Images might be too blurry or have insufficient texture")
        else:
            print("  ✓ Sufficient features for matching")
        
        # Match features
        if desc_left is not None and desc_right is not None and len(kp_left) >= 2 and len(kp_right) >= 2:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = matcher.knnMatch(desc_left, desc_right, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            print(f"\nFeature matching:")
            print(f"  Matches found: {len(good_matches)}")
            
            if len(good_matches) > 0:
                # Analyze disparity
                disparities = []
                for m in good_matches:
                    x_left = kp_left[m.queryIdx].pt[0]
                    x_right = kp_right[m.trainIdx].pt[0]
                    disp = x_left - x_right
                    disparities.append(disp)
                
                disparities = np.array(disparities)
                valid_disparities = disparities[disparities > 0]
                
                if len(valid_disparities) > 0:
                    print(f"  Disparities (pixels):")
                    print(f"    Min: {valid_disparities.min():.2f}")
                    print(f"    Max: {valid_disparities.max():.2f}")
                    print(f"    Mean: {valid_disparities.mean():.2f}")
                    print(f"    Std: {valid_disparities.std():.2f}")
                    
                    if valid_disparities.std() < 2:
                        print("  ✓ Good alignment - low disparity variance")
                    else:
                        print("  ⚠ WARNING: High disparity variance")
                        print("  Images may not be properly rectified")
                else:
                    print("  ⚠ WARNING: No valid disparities found")
                    print("  Check image alignment and camera configuration")
            else:
                print("  ⚠ WARNING: No matching features found!")
                print("  Images may be too different or not from the same scene")
        
        print("\n" + "="*60 + "\n")
    
    @staticmethod
    def test_distance_calculation(baseline_mm, focal_length_px, disparity_px):
        """
        Test distance calculation with given parameters
        
        Args:
            baseline_mm: Baseline distance in mm
            focal_length_px: Focal length in pixels
            disparity_px: Disparity in pixels
        """
        print("\n" + "="*60)
        print("DISTANCE CALCULATION TEST")
        print("="*60)
        
        print(f"\nParameters:")
        print(f"  Baseline: {baseline_mm} mm")
        print(f"  Focal length: {focal_length_px} pixels")
        print(f"  Disparity: {disparity_px} pixels")
        
        if disparity_px <= 0:
            print("  ERROR: Disparity must be positive")
            return None
        
        distance = (baseline_mm * focal_length_px) / disparity_px
        
        print(f"\nCalculation:")
        print(f"  Distance = (Baseline × Focal_Length) / Disparity")
        print(f"  Distance = ({baseline_mm} × {focal_length_px}) / {disparity_px}")
        print(f"  Distance = {distance:.2f} mm")
        print(f"  Distance = {distance/1000:.3f} m")
        
        print(f"\n" + "="*60 + "\n")
        
        return distance
    
    @staticmethod
    def camera_parameter_guide():
        """
        Print guide for determining camera parameters
        """
        print("\n" + "="*60)
        print("CAMERA PARAMETER DETERMINATION GUIDE")
        print("="*60)
        
        guide = """
BASELINE (Distance between cameras):
  1. Physically measure the distance between your camera lenses (mm)
  2. Be precise - this directly affects distance accuracy
  3. Common values: 60mm (compact), 120mm (standard), 240mm (wide baseline)

FOCAL LENGTH (in pixels):
  Method 1 - From camera specs:
    - Find your camera's focal length in mm (f)
    - Find sensor width in mm (S)
    - Focal length in pixels = (image_width_pixels × f) / S
  
  Method 2 - From calibration:
    - Use stereo_calibration.py with checkerboard images
    - This gives the most accurate focal length
  
  Method 3 - Rough estimate:
    - Typical values: 300-1000 pixels
    - Smaller sensors/higher zoom = higher focal length

DISPARITY RANGE:
  - Larger baseline = larger disparity for same object
  - Higher resolution images = more detailed disparity
  - Far objects = smaller disparity
  - Close objects = larger disparity

EXAMPLES:
  Standard setup (60-120mm baseline, normal lens):
    - Focal length: 500-800 pixels
    - Disparity range: 10-200 pixels
    - Good for: 0.5m to 10m distances
  
  Wide baseline (240mm baseline):
    - Focal length: 800-1200 pixels
    - Disparity range: 20-300 pixels
    - Good for: Far distance measurements

VALIDATION:
  1. Run stereo_vision_basic.py with test images
  2. Check if distances match reality
  3. If distances are too small: increase baseline or focal_length
  4. If distances are too large: decrease baseline or focal_length
  5. Adjust by roughly the factor of error
        """
        
        print(guide)
        print("="*60 + "\n")
    
    @staticmethod
    def interactive_parameter_tuner():
        """
        Interactive tool to find best parameters
        """
        print("\n" + "="*60)
        print("INTERACTIVE PARAMETER TUNER")
        print("="*60)
        
        print("\nThis tool helps you find the right camera parameters.")
        print("You'll need a stereo image pair with known object distances.\n")
        
        try:
            # Get image paths
            left_path = input("Left image path: ").strip()
            right_path = input("Right image path: ").strip()
            
            # Load images
            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)
            
            if img_left is None or img_right is None:
                print("Error: Could not load images")
                return
            
            # Get known values
            print("\nEnter a known distance measurement from your images:")
            print("(Example: If you know an object is 2 meters away and")
            print(" its bounding box center disparity is 50 pixels)")
            
            known_distance_m = float(input("Known distance (meters): "))
            known_disparity = float(input("Measured disparity (pixels): "))
            
            known_distance_mm = known_distance_m * 1000
            
            if known_disparity <= 0:
                print("Error: Disparity must be positive")
                return
            
            # Estimate baseline and focal length
            print("\nEnter your baseline estimate (mm):")
            baseline = float(input("Baseline (mm) [default 120]: ") or "120")
            
            # Calculate focal length
            focal_length = (baseline * known_disparity) / known_distance_mm
            
            print(f"\n" + "="*60)
            print("ESTIMATED PARAMETERS")
            print("="*60)
            print(f"  Baseline: {baseline} mm")
            print(f"  Focal Length: {focal_length:.1f} pixels")
            print(f"\nUse these values in stereo_vision_basic.py and stereo_vision_sift.py")
            print("="*60 + "\n")
            
        except ValueError:
            print("Invalid input")
    
    @staticmethod
    def print_menu():
        """Print interactive menu"""
        menu = """
STEREO VISION DIAGNOSTICS MENU
==============================

1. Check image alignment
2. Test distance calculation
3. Camera parameter guide
4. Interactive parameter tuner
5. Estimate focal length
6. Exit

"""
        print(menu)


def main():
    print("\n" + "="*60)
    print("STEREO VISION DIAGNOSTICS TOOL")
    print("="*60)
    
    while True:
        StereoDiagnostics.print_menu()
        choice = input("Select option (1-6): ").strip()
        
        if choice == "1":
            left_path = input("Left image path: ").strip()
            right_path = input("Right image path: ").strip()
            StereoDiagnostics.check_image_alignment(left_path, right_path)
        
        elif choice == "2":
            try:
                baseline = float(input("Baseline (mm): "))
                focal_length = float(input("Focal length (pixels): "))
                disparity = float(input("Disparity (pixels): "))
                StereoDiagnostics.test_distance_calculation(baseline, focal_length, disparity)
            except ValueError:
                print("Invalid input")
        
        elif choice == "3":
            StereoDiagnostics.camera_parameter_guide()
        
        elif choice == "4":
            StereoDiagnostics.interactive_parameter_tuner()
        
        elif choice == "5":
            try:
                image_path = input("Image path: ").strip()
                known_distance = float(input("Known distance (mm): "))
                known_size = float(input("Object size in image (pixels): "))
                StereoDiagnostics.estimate_focal_length_from_images(image_path, known_distance, known_size)
            except ValueError:
                print("Invalid input")
        
        elif choice == "6":
            print("Exiting...")
            break
        
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
