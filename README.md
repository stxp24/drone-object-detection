# Drone-Based Object Detection and Tracking (Simulated)
A real-time object detection and tracking system for aerial surveillance using YOLOv11, designed for detecting and tracking vehicles (civilian/military), aircraft, and maritime vessels from drone footage.
# Overview
This system implements state-of-the-art object detection using the YOLOv11 model, trained on aerial imagery datasets, achieving high-accuracy real-time detection across 6 object classes with comprehensive tracking capabilities.
# Key Features
- Real-time object detection at 30+ FPS
- Multi-object tracking using [ByteTrack](https://blog.roboflow.com/what-is-bytetrack-computer-vision/) algorithm
- 6-class detection: Cars, Buses, Trucks, Boats, Aircraft, Military Vehicles
- Comprehensive evaluation framework with per-class metrics
- Production-ready inference pipeline with visualization dashboard (not yet implemented)
# Tech Stack
- Framework: Ultralytics YOLOv11
- Model: YOLOv11-Small (optimized for accuracy/speed balance)
- Training Platform: Google Colab (T4 GPU)
- Languages: Python 3.10+
- Key Libraries: PyTorch, OpenCV, NumPy
# Dataset
- VisDrone2019: 7,000+ drone-captured images
- Custom Dataset: 134 manually annotated images (via DroneStock, annotated with CVAT)
- Total: ~7,200 images
# Training Platform
- Hardware: Google Colab T4 GPU (16GB VRAM)
- Duration: 2 hours
- Framework: Ultralytics 8.3.x
# RESULTS
### Overall Performance
Metric | Score
| mAP@50 | 80.3% |
| mAP@50-95 | 62.4% |
| Precision | 88.0% |
| Recall | 75.7% |  
### Per-class Performance
Car: 84.0% mAP@50
Bus: 60.5% mAP@50
Truck: 38.9% mAP@50
Boat: 99.5% mAP@50
Aircraft: 99.5% mAP@50
Military: 99.5% mAP@50
# Dataset Citation
@ARTICLE{9573394,
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Detection and Tracking Meet Drones Challenge},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3119563}}
