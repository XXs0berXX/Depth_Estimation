"""
Stereo Camera Calibration Tool
Use this script to calibrate your stereo camera setup
Generates calibration parameters for use in stereo_vision_basic.py and stereo_vision_sift.py
"""

import cv2
import numpy as np
from pathlib import Path

class StereoCameraCalibration:
    """
    Calibrate stereo camera pair using checkerboard pattern
    """
    
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025):
        """
        Args:
            checkerboard_size: (width, height) of checkerboard (number of corners)
            square_size: Size of each square in meters
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        self.objpoints = []  # 3d points
        self.imgpoints_left = []  # 2d points in left image
        self.imgpoints_right = []  # 2d points in right image
    
    def find_checkerboard(self, img):
        """
        Find checkerboard corners in image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            # Refine corner positions
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return True, corners_refined
        
        return False, None
    
    def calibrate_from_images(self, left_img_dir, right_img_dir):
        """
        Calibrate from directory of left and right images
        
        Args:
            left_img_dir: Directory containing left camera images
            right_img_dir: Directory containing right camera images
        """
        left_path = Path(left_img_dir)
        right_path = Path(right_img_dir)
        
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        left_images = sorted([f for f in left_path.glob('*') 
                             if f.suffix.lower() in img_extensions])
        right_images = sorted([f for f in right_path.glob('*') 
                              if f.suffix.lower() in img_extensions])
        
        if len(left_images) != len(right_images):
            print(f"Warning: Different number of images in left ({len(left_images)}) "
                  f"and right ({len(right_images)}) directories")
        
        num_images = min(len(left_images), len(right_images))
        print(f"Found {num_images} image pairs for calibration")
        
        for i, (left_img_path, right_img_path) in enumerate(zip(left_images[:num_images], 
                                                                   right_images[:num_images])):
            print(f"Processing image pair {i+1}/{num_images}...", end=' ')
            
            img_left = cv2.imread(str(left_img_path))
            img_right = cv2.imread(str(right_img_path))
            
            if img_left is None or img_right is None:
                print("Failed to load")
                continue
            
            ret_left, corners_left = self.find_checkerboard(img_left)
            ret_right, corners_right = self.find_checkerboard(img_right)
            
            if ret_left and ret_right:
                self.objpoints.append(self.objp)
                self.imgpoints_left.append(corners_left)
                self.imgpoints_right.append(corners_right)
                print("OK")
            else:
                print("Checkerboard not found")
        
        print(f"\nSuccessfully detected checkerboard in {len(self.objpoints)} image pairs")
        
        if len(self.objpoints) < 3:
            print("Error: Need at least 3 valid image pairs for calibration")
            return None
        
        return self.calibrate()
    
    def calibrate_from_video(self, left_video, right_video, frame_interval=10):
        """
        Calibrate from video files
        
        Args:
            left_video: Path to left camera video
            right_video: Path to right camera video
            frame_interval: Process every nth frame
        """
        cap_left = cv2.VideoCapture(left_video)
        cap_right = cv2.VideoCapture(right_video)
        
        frame_count = 0
        success_count = 0
        
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            
            if not ret_left or not ret_right:
                break
            
            frame_count += 1
            
            if frame_count % frame_interval == 0:
                ret_left, corners_left = self.find_checkerboard(frame_left)
                ret_right, corners_right = self.find_checkerboard(frame_right)
                
                if ret_left and ret_right:
                    self.objpoints.append(self.objp)
                    self.imgpoints_left.append(corners_left)
                    self.imgpoints_right.append(corners_right)
                    success_count += 1
                    print(f"Found checkerboard - Total: {success_count}")
        
        cap_left.release()
        cap_right.release()
        
        print(f"Successfully detected checkerboard in {success_count} frames")
        
        if len(self.objpoints) < 3:
            print("Error: Need at least 3 valid frames for calibration")
            return None
        
        return self.calibrate()
    
    def calibrate(self):
        """
        Perform stereo calibration
        """
        print("\nPerforming stereo calibration...")
        
        # Get image size from first image point
        img_size = (640, 480)  # Default, will be overridden
        
        # Calibrate left camera
        ret_left, K_left, D_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, img_size, None, None
        )
        
        # Calibrate right camera
        ret_right, K_right, D_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, img_size, None, None
        )
        
        # Stereo calibration
        ret_stereo, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            K_left, D_left, K_right, D_right, img_size
        )
        
        if ret_stereo:
            print("Stereo calibration successful!")
        else:
            print("Warning: Stereo calibration encountered issues")
        
        return {
            'K_left': K_left,
            'D_left': D_left,
            'K_right': K_right,
            'D_right': D_right,
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'img_size': img_size
        }


def print_calibration_results(calib_data):
    """Pretty print calibration results"""
    
    print("\n" + "="*60)
    print("STEREO CALIBRATION RESULTS")
    print("="*60)
    
    print("\nLeft Camera Matrix (K_left):")
    print(calib_data['K_left'])
    
    print("\nLeft Distortion Coefficients (D_left):")
    print(calib_data['D_left'])
    
    print("\nRight Camera Matrix (K_right):")
    print(calib_data['K_right'])
    
    print("\nRight Distortion Coefficients (D_right):")
    print(calib_data['D_right'])
    
    print("\nRotation Matrix (R) - Left to Right:")
    print(calib_data['R'])
    
    print("\nTranslation Vector (T) - Left to Right:")
    print(calib_data['T'])
    print(f"Baseline (distance between cameras): {np.linalg.norm(calib_data['T']):.3f} units")
    
    print("\n" + "="*60)
    print("Update the parameters in stereo_vision_basic.py and")
    print("stereo_vision_sift.py with these values")
    print("="*60 + "\n")


def main():
    print("Stereo Camera Calibration Tool")
    print("="*60)
    
    print("\nOptions:")
    print("1. Calibrate from image directories")
    print("2. Calibrate from video files")
    print("3. Manual calibration parameters (skip calibration)")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    calibrator = StereoCameraCalibration(
        checkerboard_size=(9, 6),  # Adjust based on your checkerboard
        square_size=0.025  # 2.5 cm squares
    )
    
    calib_data = None
    
    if choice == "1":
        left_dir = input("Enter path to left camera images directory: ").strip()
        right_dir = input("Enter path to right camera images directory: ").strip()
        calib_data = calibrator.calibrate_from_images(left_dir, right_dir)
    
    elif choice == "2":
        left_video = input("Enter path to left camera video: ").strip()
        right_video = input("Enter path to right camera video: ").strip()
        frame_interval = int(input("Enter frame interval (e.g., 10): ").strip() or "10")
        calib_data = calibrator.calibrate_from_video(left_video, right_video, frame_interval)
    
    elif choice == "3":
        print("\nUsing default calibration parameters")
        print("You can manually edit stereo_vision_basic.py and stereo_vision_sift.py")
        print("to set your camera parameters")
        return
    
    if calib_data:
        print_calibration_results(calib_data)
        
        # Save to file
        output_path = Path("calibration_results.txt")
        with open(output_path, 'w') as f:
            f.write("STEREO CALIBRATION RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"K_left:\n{calib_data['K_left']}\n\n")
            f.write(f"D_left:\n{calib_data['D_left']}\n\n")
            f.write(f"K_right:\n{calib_data['K_right']}\n\n")
            f.write(f"D_right:\n{calib_data['D_right']}\n\n")
            f.write(f"R:\n{calib_data['R']}\n\n")
            f.write(f"T:\n{calib_data['T']}\n\n")
        
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
