# Drone-Based Object Detection and Tracking (Simulated)
A real-time object detection and tracking system for aerial surveillance using YOLOv11, designed for detecting and tracking vehicles (civilian/military), aircraft, and maritime vessels from drone footage.
# Overview
This system implements state-of-the-art object detection using the YOLOv11 model, trained on aerial imagery datasets, hoping to achieve high-accuracy real-time detection across 6 object classes with comprehensive tracking capabilities.
# Key Features
- Real-time object detection at 30+ FPS
- Multi-object tracking using [ByteTrack](https://blog.roboflow.com/what-is-bytetrack-computer-vision/) algorithm
- 6-class detection: Cars, Buses, Trucks, Boats, Aircraft, Military Vehicles
- Comprehensive evaluation framework with per-class metrics
- Production-ready inference pipeline with visualization dashboard (not yet implemented)
# Tech Stack
Framework: Ultralytics YOLOv11
Model: YOLOv11-Small (optimized for accuracy/speed balance)
Training Platform: Google Colab (T4 GPU)
Languages: Python 3.10+
Key Libraries: PyTorch, OpenCV, NumPy
# Dataset
VisDrone2019: 7,000+ drone-captured images
Custom Dataset: 134 manually annotated images
Total: ~7,200 images with 50,000+ annotations
# Training Platform
Hardware: Google Colab T4 GPU (16GB VRAM)
Duration: In Progress
Framework: Ultralytics 8.3.x
# THIS PROJECT IS CURRENTLY IN TRAINING
This project is not yet finished, as training is currently running on my Colab project. This README.md, as well as project structure, data, etc will be updated as this project progresses. I may edit this README.md sporadically, such as without any code pushes or data pushes. Sometimes I just don't like some things I have written and will change it. For any questions, email me at dnc8878@g.rit.edu
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
