### Stereo Vision with YOLOv8 - Complete Guide

This folder contains three stereo vision scripts for 3D object detection and distance estimation.

---

## Files Included

### 1. **stereo_vision_basic.py** - Basic Stereo Vision
- Uses YOLOv8 for object detection
- Calculates distance based on disparity (difference in x-coordinates)
- Simple, fast, and easy to understand
- **Best for:** Quick testing, known camera parameters

### 2. **stereo_vision_sift.py** - SIFT-Enhanced Stereo Vision
- Combines YOLOv8 detection with SIFT feature matching
- More robust feature correspondence between left and right images
- Better handling of image variations (scale, rotation, lighting)
- **Best for:** More accurate distance estimation, challenging conditions

### 3. **stereo_calibration.py** - Camera Calibration Tool
- Calibrate your stereo camera pair
- Generates intrinsic and extrinsic camera parameters
- Supports calibration from images or videos
- **Required for:** Accurate distance measurements with your specific camera

---

## Quick Start

### Step 1: Prepare Your Stereo Images

Place two images (left and right views of the same scene) in the MS2 folder:
- `stereo_left.jpg` - Left camera image
- `stereo_right.jpg` - Right camera image

**Important:** 
- Images should be roughly synchronized (same moment in time)
- For best results, use stereo pairs with known baseline distance between cameras
- Keep the same resolution and camera orientation

### Step 2: Configure Camera Parameters

Edit the camera parameters in the script:

```python
# BASELINE = distance between your left and right cameras (in mm)
BASELINE = 120  # Change this to your camera setup

# FOCAL_LENGTH = focal length in pixels
FOCAL_LENGTH = 800  # Adjust based on your camera
```

**If you don't know these values**, use `stereo_calibration.py`:

```bash
python stereo_calibration.py
```

### Step 3: Run Stereo Vision

#### Option A: Basic Version (Fast)
```bash
python stereo_vision_basic.py
```

#### Option B: SIFT-Enhanced (More Accurate)
```bash
python stereo_vision_sift.py
```

### Output

The script will:
1. **Display** detected objects with distance measurements
2. **Print** detailed information to console:
   ```
   Object 1:
     Class: car
     Confidence: 0.95
     Disparity: 45.3 pixels
     Distance: 2133.33 mm (2.13 m)
     Position: (320, 240)
   ```
3. **Save** result image showing both stereo views with bounding boxes

---

## Camera Calibration

### Why Calibrate?

Calibration improves distance accuracy by:
- Accounting for lens distortion
- Measuring true focal length for your camera
- Precise baseline distance measurement

### How to Calibrate

#### Method 1: From Images

1. Print a checkerboard pattern (9x6 recommended)
   - Each square should be 2.5cm x 2.5cm
   - OR adjust square_size in the code

2. Capture 20-30 stereo image pairs showing the checkerboard at different angles
   ```
   left_calib/
     image_1.jpg
     image_2.jpg
     ...
   
   right_calib/
     image_1.jpg
     image_2.jpg
     ...
   ```

3. Run calibration:
   ```bash
   python stereo_calibration.py
   # Select option 1
   # Enter left and right image directories
   ```

#### Method 2: From Video

1. Record stereo video of checkerboard at various angles
   ```bash
   python stereo_calibration.py
   # Select option 2
   # Enter left and right video file paths
   ```

### Using Calibration Results

After calibration, update these parameters in both stereo scripts:

```python
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

D = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

BASELINE = baseline_distance_mm
```

---

## API Reference

### StereoVisionDetector (Basic)

```python
detector = StereoVisionDetector(model_path, baseline, focal_length)

# Detect objects
detections = detector.detect_objects(image)

# Calculate distance
dist_info = detector.calculate_disparity_and_distance(bbox_left, bbox_right)

# Match and measure
matches = detector.match_bounding_boxes(detections_left, detections_right)

# Draw results
result = detector.draw_results(img_left, img_right, matches)
```

### StereoVisionSIFT (Enhanced)

```python
stereo = StereoVisionSIFT(model_path, baseline, focal_length)

# Detect objects
detections = stereo.detect_objects(image)

# Extract SIFT features from bounding box
kp, desc = stereo.extract_bbox_features(image, bbox)

# Match and measure with SIFT
measurements = stereo.match_and_measure_stereo(img_left, img_right, 
                                                detections_left, detections_right)

# Draw results
result = stereo.draw_results(img_left, img_right, measurements)
```

---

## Troubleshooting

### Issue: No objects detected

**Solution:**
- Objects might be too small or too far away
- Adjust YOLOv8 confidence threshold in the script
- Ensure images are clear and well-lit

### Issue: Matching fails / No distance calculated

**Solutions:**
- **Objects not at similar y-coordinates:** Adjust `max_y_diff` parameter
- **Very different perspectives:** Try SIFT version (`stereo_vision_sift.py`)
- **Improper baseline calibration:** Remeasure or recalibrate cameras

### Issue: Distances seem incorrect

**Solutions:**
1. **Verify camera calibration** - Run `stereo_calibration.py`
2. **Check focal length** - Should be around 300-1000 pixels for typical cameras
3. **Verify baseline** - Measure distance between cameras in mm
4. **Ensure proper image alignment** - Left and right images should be rectified

### Issue: SIFT version is slow

**Solution:**
- Reduce image resolution
- Use fewer features: Modify `self.sift.detectAndCompute()` parameters
- Use basic version for real-time applications

---

## Understanding the Output

### Distance = (Baseline × Focal_Length) / Disparity

- **Baseline**: Distance between left and right cameras (mm)
- **Focal_Length**: Focal length in pixels
- **Disparity**: Difference in x-coordinates of matched features (pixels)

**Example:**
```
Baseline = 120 mm
Focal Length = 800 pixels
Disparity = 45 pixels

Distance = (120 × 800) / 45 = 2133 mm ≈ 2.1 meters
```

---

## Advanced Usage

### Modify Detection Classes

Edit the YOLOv8 model or class filtering:

```python
# In detect_objects():
for box in results.boxes:
    cls_id = int(box.cls)
    # Filter specific classes (e.g., only cars)
    if cls_id == 2:  # 2 = car in COCO
        detections.append(...)
```

### Batch Processing

```python
from pathlib import Path

stereo_pairs = [
    ("img1_left.jpg", "img1_right.jpg"),
    ("img2_left.jpg", "img2_right.jpg"),
    # ...
]

for left, right in stereo_pairs:
    img_l = cv2.imread(left)
    img_r = cv2.imread(right)
    # Run detection...
```

### Real-time Stereo from Cameras

```python
cap_left = cv2.VideoCapture(0)   # Left camera
cap_right = cv2.VideoCapture(1)  # Right camera

while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    
    if ret_l and ret_r:
        detections_l = detector.detect_objects(frame_l)
        detections_r = detector.detect_objects(frame_r)
        matches = detector.match_bounding_boxes(detections_l, detections_r)
        # Process...
```

---

## Common Parameters

### Fine-tuning Detection

```python
# In detect_objects() method:
results = self.model(image, conf=0.5)  # Confidence threshold
# Lower conf = more detections but more false positives
# Higher conf = fewer detections but higher accuracy
```

### Fine-tuning Matching

```python
# In match_bounding_boxes():
max_y_diff = 50  # Pixels - tolerance for y-coordinate difference
# Higher = more permissive matching
# Lower = more strict matching
```

### SIFT Parameters (Advanced)

```python
# In StereoVisionSIFT.__init__():
self.sift = cv2.SIFT_create()  # Use default parameters

# Or customize:
self.sift = cv2.SIFT_create(
    nfeatures=0,
    nOctaveLayers=3,
    contrastThreshold=0.03,
    edgeThreshold=10,
    sigma=1.6
)
```

---

## Performance Tips

1. **Resize images** to smaller resolution for faster processing
   ```python
   img = cv2.resize(img, (640, 480))
   ```

2. **Use GPU** for YOLOv8:
   ```python
   model = YOLO("yolov8n.pt")
   model.to('cuda')
   ```

3. **Batch processing**: Process multiple image pairs together

4. **Choose lighter model**: `yolov8n.pt` (nano) is faster than `yolov8m.pt` (medium)

---

## Dependencies

- opencv-python
- numpy
- ultralytics (YOLOv8)

Install with:
```bash
pip install opencv-python numpy ultralytics
```

---

## Notes

- The basic version assumes cameras are properly rectified (epipolar geometry satisfied)
- SIFT version is more robust but slower
- For best results, use synchronized stereo cameras with known calibration
- Keep baseline distance in same units for consistency

---

## Questions & Support

If you encounter issues:
1. Check camera calibration parameters
2. Verify image pair quality and synchronization
3. Ensure YOLOv8 model is detecting objects correctly
4. Try SIFT version for difficult cases
5. Adjust parameters (baseline, focal_length, max_y_diff)
