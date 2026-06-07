"""
SixRay v3 - Image Detection
============================
Paste your image path below and run the script.
Draws green bounding boxes on detected objects and saves the result.
"""

import cv2
from pathlib import Path
from ultralytics import YOLO



# D:\archive\sixray_v3\testinging\one.png
IMAGE_PATH = r"D:\archive\sixray_v3\testinging\one.png"  
CONFIDENCE = 0.25
# ──────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS = PROJECT_DIR / "runs" / "detect" / "sixray_model" / "weights" / "best.pt"

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2


print(f"Loading model...")
model = YOLO(str(WEIGHTS))
class_names = model.names


image_path = Path(IMAGE_PATH)
frame = cv2.imread(str(image_path))
if frame is None:
    print(f"ERROR: Could not read image: {image_path}")
    exit()

# Convert to black & white before detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

print(f"Scanning (B/W): {image_path}")

# Run detection
results = model(frame, conf=CONFIDENCE, verbose=False)

# Draw green boxes
for result in results:
    if result.boxes is None:
        continue
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls_id]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1, y1 - 5), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
        print(f"  Found: {class_names[cls_id]} ({conf:.1%})")

# Save result
output_path = image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
cv2.imwrite(str(output_path), frame)
print(f"\nSaved: {output_path}")
