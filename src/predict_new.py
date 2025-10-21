# test_on_video.py
from ultralytics import YOLO
import cv2
from pathlib import Path

# Load your trained model
model = YOLO("models/colab-best.pt")

# video path
video_path = r"src/videos/drone-view-of-la-traffic-8211;-134-and-5-freeway-interchange-in-rush-hour.mp4"
output_path = r"src/videos/output_final.mp4"

# Get video properties
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Process video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Processing video...")
frame_count = 0

for result in model.track(
        source=video_path,
        conf=0.25,
        iou=0.5,
        show=True,
        stream=True,
        tracker="bytetrack.yaml",
        verbose=False
):
    annotated_frame = result.plot()
    out.write(annotated_frame)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

out.release()
cv2.destroyAllWindows()

print(f"Output: {output_path}")
