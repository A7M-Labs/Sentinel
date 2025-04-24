

# Methodology

This document describes the datasets and test inputs used, the hardware and software environment, and the experimental procedures followed to benchmark and validate the real-time security detection pipeline.

## 1. Data

- **Source Videos**  
  We used a mix of pre-recorded MP4 clips stored under `videos/`, representing both normal and “robbery” scenarios:
  - Normal: `normal1.mp4` … `normal5.mp4` (640×360 at 15–30 FPS)
  - Robbery: `rob1.mp4` … `rob5.mp4` (640×360 at 15–30 FPS)
- **Segment Length**  
  Each clip was broken into 5 s segments (75 frames at 15 FPS) for search API queries and change detection.
- **Ground Truth**  
  For evaluation, manually annotated event timestamps (start/end) were created for a subset of clips to measure recall and precision.

## 2. Hardware & System

- **CPU**: Intel Core i7-13700K (16 cores, 24 threads)  
- **Memory**: 32 GB DDR5 RAM  
- **Storage**: Samsung 990 Pro 1 TB NVMe SSD  
- **GPU**: NVIDIA GeForce RTX 4080 Super  
- **Operating System**: Ubuntu 22.04 LTS (Linux kernel 5.15)

## 3. Software Stack

- **Language**: Python 3.11  
- **Deep Learning Framework**: PyTorch 2.x with CUDA 12.x support  
- **Object Detection**: Ultralytics YOLOv8 (`yolov8n.pt`)  
- **Computer Vision**: OpenCV 4.7.0  
- **Web UI**: Streamlit 1.25.x, streamlit_drawable_canvas  
- **API Client**: TwelveLabs Python SDK (`twelvelabs==1.x`)  
- **Database**: SQLite 3.36 (WAL mode)  
- **Concurrency**: `concurrent.futures.ThreadPoolExecutor`, `threading`  
- **Assembly JIT**: `ctypes`, `mmap` (Linux) / `VirtualAlloc` (Windows)  
- **Utilities**: `hashlib`, `numpy`, `pathlib`