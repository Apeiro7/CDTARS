# 🚦 CDTARS – Collision Detection & Traffic Analysis in Real-Time Systems

![GitHub stars](https://img.shields.io/github/stars/Apeiro7/CDTARS?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/Apeiro7/CDTARS?style=for-the-badge)
![GitHub license](https://img.shields.io/github/license/Apeiro7/CDTARS?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research%20Project-green?style=for-the-badge)

---

## 📄 Abstract

**CDTARS (Collision Detection & Traffic Analysis in Real-Time Systems)** is an intelligent traffic monitoring framework that integrates **YOLOv8-based object detection**, **multi-object tracking**, and **custom-trained accident detection models** for real-time analysis of road environments.

The system enables accurate vehicle detection, trajectory tracking, and abnormal event identification, thereby enhancing **traffic efficiency**, **road safety**, and **emergency response mechanisms**. CDTARS contributes toward the development of **smart cities** and **intelligent transportation systems (ITS)**.

---

## 🔑 Keywords

YOLOv8, Computer Vision, Traffic Analysis, Accident Detection, Deep Learning, Object Tracking, Intelligent Transportation Systems (ITS)

---

## 1️⃣ Introduction

With the rapid growth of urban transportation systems, traditional traffic monitoring approaches fail to provide real-time insights and automated decision-making.

CDTARS addresses these challenges by:
- Automating vehicle detection using deep learning  
- Tracking vehicle movement across frames  
- Detecting accidents and abnormal behavior  
- Providing actionable insights for traffic management  

---

## 📌 Features

- 🚗 Real-Time Vehicle Detection  
- 🧠 Custom Accident Detection Model  
- 🎯 High-Accuracy YOLOv8 Object Detection  
- 📍 Vehicle Tracking with ByteTrack  
- 📊 Traffic Flow Analysis  
- ⚠️ Abnormal Event Recognition (Accidents)  
- 🎥 Video Processing & Annotation  
- ⚡ GPU-Accelerated Inference  

---

## 🏗️ System Architecture

```
Input Video Stream  
      ↓  
YOLOv8 Detection  
      ↓  
Object Tracking (ByteTrack)  
      ↓  
Event Analysis  
      ↓  
Accident Detection  
      ↓  
Annotated Output + Insights  
```

---

## 2️⃣ Methodology

### 2.1 Vehicle Detection

```python
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.fuse()
```

Supported classes:
- Car  
- Motorcycle  
- Bus  
- Truck  

---

### 2.2 Object Tracking

```python
byte_tracker = sv.ByteTrack()
```

- Assigns unique IDs  
- Tracks vehicle trajectories  
- Handles multi-object tracking  

---

### 2.3 Accident Detection

```python
from ultralytics import YOLO

model = YOLO("/content/best.pt")
model.fuse()
```

- Custom-trained model (Roboflow dataset)  
- Detects collisions and anomalies  
- Enables real-time alerts  

---

### 2.4 Video Processing Pipeline

- Frame-by-frame inference  
- Annotation using Supervision  
- Output video generation  

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/CDTARS.git
cd CDTARS
pip install -r requirements.txt
```

(Optional GPU check)
```bash
nvidia-smi
```

---

## 🚀 Usage

Run Jupyter Notebook:

```bash
jupyter notebook CDTARS.ipynb
```

Or run script:

```bash
python detect.py --input sample_video.mp4
```

---

## 🎥 Demo

### Sample Detection Output
<img src="assets/output1.png" width="600"/>

---

## 📊 Results & Visualization

### Vehicle Count Analysis
<img src="assets/output4.jpeg" width="600"/>

### Detection Confidence & Performance
<img src="assets/output2.png" width="600"/>
<img src="assets/output3.png" width="600"/>

---

## 📊 Results & Observations

| Metric                  | Performance |
|------------------------|------------|
| Detection Accuracy     | High       |
| Real-time Processing   | Yes (GPU)  |
| Tracking Stability     | High       |
| Accident Detection     | Effective  |

---

## ⚡ Challenges Addressed

- Limited labeled accident datasets  
- Real-time processing constraints  
- Multi-object tracking complexity  
- Hardware resource limitations  

---

## 🔮 Future Improvements

- Integration with **5G / V2X communication**  
- Reinforcement Learning for adaptive signals  
- Edge deployment (Jetson, Raspberry Pi)  
- Cloud-based analytics dashboard  
- Multi-camera fusion systems  

---

## 🚀 Applications

- Smart City Traffic Management  
- Highway Surveillance Systems  
- Accident Detection Systems  
- Autonomous Traffic Monitoring  
- Emergency Response Optimization  

---

## 📈 Impact

- 🚫 Reduced road accidents  
- 🚦 Improved traffic flow  
- 🚑 Faster emergency response  
- 🌐 Smarter infrastructure  

---

## 🛠️ Technologies Used

| Technology   | Role                     |
|-------------|--------------------------|
| YOLOv8      | Object Detection         |
| Supervision | Annotation & Tracking    |
| ByteTrack   | Multi-object Tracking    |
| Roboflow    | Dataset Management       |
| OpenCV      | Video Processing         |
| Python      | Core Implementation      |

---

## 🤝 Contributing

```bash
fork → create branch → commit → pull request
```

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**Amit Bhardwaj**  
B.Tech CSE | AI & Traffic Systems Researcher  

---

## ⭐ Support

If you found this project useful, please give it a **star ⭐ on GitHub**!
