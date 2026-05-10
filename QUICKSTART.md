## Stereo Vision with YOLOv8 - MS2 Implementation

**Complete stereo vision package with bounding box distance estimation and SIFT enhancement**

---

## 📋 Files Created

### Core Scripts

| File | Purpose | Use When |
|------|---------|----------|
| **stereo_vision_basic.py** | Basic stereo vision with YOLOv8 | You want fast, simple distance measurements |
| **stereo_vision_sift.py** | SIFT-enhanced stereo vision | You need more accurate/robust matching |
| **stereo_calibration.py** | Camera calibration tool | You need to calibrate your camera pair |
| **stereo_diagnostics.py** | Diagnostic & parameter tuning | You need to find the right camera parameters |
| **example_usage.py** | Ready-to-use examples | You want to see working code examples |

### Documentation

| File | Content |
|------|---------|
| **README_STEREO_VISION.md** | Complete guide with all details |
| **QUICKSTART.md** | This file - quick reference |

---

## 🚀 Quick Start (5 minutes)

### 1. Prepare Your Images

Place stereo image pair in MS2 folder:
```
stereo_left.jpg   (left camera view)
stereo_right.jpg  (right camera view)
```

### 2. Update Camera Parameters

Edit one of these scripts and set:
```python
BASELINE = 120        # Distance between cameras (mm)
FOCAL_LENGTH = 800    # Focal length (pixels)
```

**Don't know your parameters?** Run:
```bash
python stereo_diagnostics.py
```

### 3. Run Stereo Vision

**Option A - Fast & Simple:**
```bash
python stereo_vision_basic.py
```

**Option B - More Accurate:**
```bash
python stereo_vision_sift.py
```

### 4. View Results

- Console output shows distance for each detected object
- Result image saved to `output/stereo/` folder
- Distances in mm and meters

---

## 📊 Example Output

```
Object 1:
  Class: car
  Confidence: 0.95
  Disparity: 45.3 pixels
  Distance: 2133.33 mm (2.13 m)
  Position: (320, 240)

Object 2:
  Class: person
  Confidence: 0.87
  Disparity: 120.5 pixels
  Distance: 800.00 mm (0.80 m)
  Position: (450, 350)
```

---

## 🔧 Finding Your Camera Parameters

### Method 1: Quick Estimate (⚡ Fast)
```bash
python stereo_diagnostics.py
# Select option 4: Interactive Parameter Tuner
# Follow on-screen instructions
```

### Method 2: Proper Calibration (⭐ Recommended)

1. Print checkerboard pattern (9x6 grid, 2.5cm squares)
2. Capture 20-30 stereo pairs of checkerboard at different angles
3. Run calibration:
```bash
python stereo_calibration.py
# Select option 1: Calibrate from images
# Enter left and right image directories
```

4. Copy generated parameters to stereo scripts

### Method 3: Manual Entry
- **Baseline:** Measure distance between camera lenses (mm)
- **Focal Length:** 300-1000 pixels (depends on camera and zoom)

---

## 🎯 Usage Examples

### Basic Usage
```python
from stereo_vision_basic import StereoVisionDetector
import cv2

# Create detector
detector = StereoVisionDetector(
    model_path="yolov8n.pt",
    baseline=120,
    focal_length=800
)

# Load images
img_left = cv2.imread("stereo_left.jpg")
img_right = cv2.imread("stereo_right.jpg")

# Detect and measure
detections_left = detector.detect_objects(img_left)
detections_right = detector.detect_objects(img_right)
matches = detector.match_bounding_boxes(detections_left, detections_right)

# Print distances
for match in matches:
    det = match['left_detection']
    dist = match['distance_info']['distance']
    print(f"{det['class_name']}: {dist/1000:.2f} m")
```

### SIFT Version
```python
from stereo_vision_sift import StereoVisionSIFT
import cv2

stereo = StereoVisionSIFT(
    model_path="yolov8n.pt",
    baseline=120,
    focal_length=800
)

img_left = cv2.imread("stereo_left.jpg")
img_right = cv2.imread("stereo_right.jpg")

detections_left = stereo.detect_objects(img_left)
detections_right = stereo.detect_objects(img_right)

# SIFT-based matching (more robust)
measurements = stereo.match_and_measure_stereo(
    img_left, img_right,
    detections_left, detections_right
)

for meas in measurements:
    det = meas['left_detection']
    dist = meas['distance']
    print(f"{det['class_name']}: {dist/1000:.2f} m")
```

---

## ⚙️ Adjustable Parameters

### YOLOv8 Detection
```python
# Confidence threshold (0-1)
# Lower = more detections, more false positives
results = model(image, conf=0.5)
```

### Bounding Box Matching
```python
# Y-coordinate tolerance (pixels)
# Higher = more lenient, may match wrong objects
# Lower = stricter, may miss valid matches
matches = detector.match_bounding_boxes(
    detections_left, 
    detections_right,
    max_y_diff=50  # Adjust this
)
```

### Distance Accuracy
```python
# Most important parameters
BASELINE = 120        # Measure precisely
FOCAL_LENGTH = 800    # Get from calibration
```

---

## 🐛 Troubleshooting

### No objects detected
- ✓ Check image quality and lighting
- ✓ Objects might be too small
- ✓ Lower YOLOv8 confidence threshold

### Matching fails (no distances calculated)
- ✓ Objects not at similar y-coordinates → increase `max_y_diff`
- ✓ Try SIFT version → more robust
- ✓ Check image alignment → run `stereo_diagnostics.py`

### Distances are wrong
- ✓ Verify baseline measurement
- ✓ Verify focal length (use calibration)
- ✓ Check image rectification
- ✓ Run diagnostics tool

### SIFT version is slow
- ✓ Reduce image resolution
- ✓ Use basic version for real-time
- ✓ Process offline or in batches

---

## 📁 Output Structure

```
output/
├── stereo/
│   └── stereo_result.jpg          (basic version results)
├── stereo_sift/
│   └── stereo_sift_result.jpg     (SIFT version results)
└── stereo_batch/
    ├── pair_1.jpg
    ├── pair_2.jpg
    └── pair_3.jpg
```

---

## 🔬 How It Works

### Basic Algorithm
1. **Detect** objects with YOLOv8 in both left and right images
2. **Match** objects by class and y-coordinate similarity
3. **Calculate disparity** = x_left - x_right (in pixels)
4. **Calculate distance** = (Baseline × Focal_Length) / Disparity

### SIFT Enhancement
1. **Detect** objects with YOLOv8
2. **Extract** SIFT features from each bounding box
3. **Match** SIFT features between left and right boxes
4. **Calculate disparity** from feature correspondences (more robust)
5. **Calculate distance** using median disparity

---

## 📚 Key Concepts

### Disparity
- Difference in x-coordinates of same object in left and right images
- Larger disparity → closer object
- Smaller disparity → farther object

### Baseline
- Physical distance between left and right camera lenses
- Larger baseline → more accurate far distances
- Smaller baseline → better for near objects

### Focal Length
- How much the camera "zooms in"
- Usually 300-1000 pixels for typical cameras
- Get from camera specs or calibration

### Rectification
- Aligning images so matching objects have same y-coordinate
- Not required if using well-aligned cameras
- Adjust `max_y_diff` if images not perfectly aligned

---

## 🎓 Next Steps

1. **Test** with your camera: Run `example_usage.py`
2. **Calibrate**: Use `stereo_calibration.py` for accuracy
3. **Optimize**: Adjust parameters using `stereo_diagnostics.py`
4. **Integrate**: Use in your project with proper parameter configuration

---

## 📞 Support

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| ImportError: No module named 'cv2' | `pip install opencv-python` |
| ImportError: No module named 'ultralytics' | `pip install ultralytics` |
| CUDA errors | Remove `.to('cuda')` or install CUDA |
| Distances unrealistic | Run calibration tool |
| Slow performance | Use basic version, reduce resolution |

### Resources

- **README_STEREO_VISION.md** - Complete documentation
- **example_usage.py** - Working code examples
- **stereo_diagnostics.py** - Parameter tuning tool

---

## 📝 License & Attribution

- Uses **YOLOv8** (Ultralytics)
- Uses **SIFT** (OpenCV)
- Compatible with OpenCV and NumPy

---

## ✅ Checklist

Before running stereo vision:

- [ ] Both stereo images placed in MS2 folder
- [ ] Images are synchronized (same moment in time)
- [ ] Camera parameters estimated or calibrated
- [ ] Updated BASELINE and FOCAL_LENGTH in script
- [ ] YOLOv8 model present (yolov8n.pt)
- [ ] Required packages installed (cv2, numpy, ultralytics)

---

**Ready to go!** Start with `python stereo_vision_basic.py` 🎯
