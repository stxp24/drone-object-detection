import cv2
from ultralytics import YOLO

import os

project_root = r"C:\Users\dncur\PycharmProjects\drone-object-detection"
model_path = os.path.join(project_root, "models", "best.pt")

# verify model exists before loading
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print(f"Current directory: {os.getcwd()}")
    exit()

model = YOLO(model_path)

# path to video
video_path = r"/src/videos/mclarens-suburban-sunset-chase-5.mp4"

# check if video exists
if not os.path.exists(video_path):
    print(f"Error: Video not found at {video_path}")
    exit()

# open video to get properties
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# define output path
output_path = r"/src/videos/mclarens-coastal-chase-1_custom_tracked.mp4"

# create video writer with H264 codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or try 'avc1' or 'X264'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# run object tracking with custom model
print("Starting object tracking with custom model...")
print(f"Processing video at {fps} FPS, resolution {width}x{height}")

# open video again for processing
cap = cv2.VideoCapture(video_path)
frame_count = 0

# process each frame with tracking
for result in model.track(
        source=video_path,
        conf=0.25,
        iou=0.5,
        show=True,
        stream=True,  # stream results frame by frame
        tracker="bytetrack.yaml",
        verbose=False  # reduce console output
):
    # get the annotated frame with bounding boxes and tracking IDs
    annotated_frame = result.plot()

    # write the annotated frame to output video
    out.write(annotated_frame)

    frame_count += 1
    if frame_count % 30 == 0:  # print progress every 30 frames
        print(f"Processed {frame_count} frames...")

print(f"\nTracking complete! Processed {frame_count} frames")
print(f"Output saved to: {os.path.abspath(output_path)}")

# release resources
cap.release()
out.release()
cv2.destroyAllWindows()
