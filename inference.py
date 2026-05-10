from ultralytics import YOLO
import cv2
from pathlib import Path

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Get COCO class index for "car" (class 2 in COCO)
CAR_CLASS_ID = 2

dataset_dir = Path("dataset")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for img_path in dataset_dir.iterdir():
    if img_path.suffix.lower() not in image_extensions:
        continue

    img = cv2.imread(str(img_path))
    results = model(img, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls)
        if cls_id != CAR_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"car {conf:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    out_path = output_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")

print("Done.")