# Instructions for Model Download
**NOTE:** The trained model weights are not included in this repository due to file size constraints.  
- Direct download link: https://drive.google.com/file/d/1vW-WOTiNb2Oj2_1xBSeMMONmyqMq7NYo/view?usp=drive_link  
- Place "best.pt" download into 'models/best.pt'.
# Details of Model Used:
- **Architecture:** YOLOv11s
- **Parameters:** 9.4M
- **Input Size:** 640x640
- **Classes:** 6 (car, bus, truck, boat, aircraft, military)
- **Dataset:** VisDrone2019 + Custom dataset (7,650 images total)
- **Training Platform:** Google Colab (for A100 GPU)
- **Training time:** ~ 2 hours (181 epochs)
- **File Size:** 18.3 MB
## Usage
```python
from ultralytics import YOLO

# Load model
model = YOLO('models/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Process video
results = model.track('path/to/video.mp4', save=True)
```
