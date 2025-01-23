# Object Detection & Situation Description in Low-Light CCTV Videos 🚦🔍

This project integrates computer vision and natural language processing to enable real-time object detection and scene summarization for low-light CCTV footage. By leveraging YOLO for object detection and a fine-tuned language model for scene descriptions, the system provides efficient and interpretable outputs for surveillance tasks.

## 📝 Table of Contents
- [Project Background](#project-background)
- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Contributors](#contributors)

---

## 🎯 Project Background
Low-light CCTV footage often poses challenges for identifying specific events in real-time. This project aims to address:
1. **Object detection in low-light conditions** using a specialized YOLO model.
2. **Scene summarization** through a fine-tuned small language model (sLLM), trained to process video tracking data into human-readable text.

---

## ✨ Features
- **YOLOv8-based Object Detection**: Real-time tracking of objects in low-light CCTV footage.
- **Video-to-Text Processing**: Scene summaries generated every 10 seconds using a fine-tuned sLLM.
- **Efficient Training Pipeline**:
  - Low-Rank Adaptation (LoRA) for cost-effective language model fine-tuning.
  - Custom training data generation using Gemini Flash.

---

## 🏗 Architecture
1. **YOLOv8 Object Detection**:
   - YOLOv8n (nano) model fine-tuned on low-light datasets.
   - Optimized for real-time performance.
2. **Video-to-Text Processing**:
   - Tracked object data includes IDs, bounding boxes, and frame intervals.
   - Summarized using Microsoft Phi-2 (2.7B parameters) fine-tuned with LoRA.
3. **Model Integration**:
   - Multiprocessing allows YOLO and sLLM to operate in parallel.
   - Summaries are generated from representative frames and stored in short-interval output files.

---

## ⚙️ Setup
### Prerequisites
- Python 3.8+
- NVIDIA GPU (minimum: RTX 3060)
- Dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lowlight-cctv.git
   cd lowlight-cctv
