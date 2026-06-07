"""
SixRay v3 - Real-Time Webcam Object Detection
===============================================
Detects: Gun, Knife, Pliers, Scissors, Wrench
Uses the trained YOLOv11 model with OpenCV webcam feed.
Green bounding boxes are drawn around detected objects.

Controls:
  - Press 'q' to quit
  - Press 's' to save a screenshot
"""

import cv2
from pathlib import Path
from ultralytics import YOLO    

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS = PROJECT_DIR / "runs" / "detect" / "sixray_model" / "weights" / "best.pt"

CONFIDENCE_THRESHOLD = 0.25   # Minimum confidence to show detection (lower = more sensitive)
CAMERA_INDEX = 0              # 0 = default webcam, 1 = external camera

# Green color for bounding boxes (BGR format)
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2


def main():
    # Load the trained model
    if not WEIGHTS.exists():
        print(f"ERROR: Model weights not found at {WEIGHTS}")
        print("Please train the model first using train_model.py")
        return

    print(f"Loading model from: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))
    class_names = model.names  # {0: 'Gun', 1: 'Knife', 2: 'Pliers', ...}
    print(f"Classes: {list(class_names.values())}")

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera (index {CAMERA_INDEX})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 50)
    print("  Webcam Detection Started")
    print("  Press 'q' to quit | Press 's' to save screenshot")
    print("=" * 50 + "\n")

    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera")
            break

        # Run YOLO inference on the frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Draw detections
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class name and confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{class_names[cls_id]} {conf:.2f}"

                # Draw green bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

                # Draw label background
                (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), BOX_COLOR, -1)

                # Draw label text (black on green background)
                cv2.putText(frame, label, (x1, y1 - 5), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

        # Show FPS
        cv2.imshow("SixRay Detection - Press 'q' to quit", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(str(PROJECT_DIR / filename), frame)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Done.")


if __name__ == "__main__":
    main()
