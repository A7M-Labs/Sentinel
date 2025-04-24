

# Benchmark Results

## 1. API Call Latency (TwelveLabs)

| Scenario                          | Naive Implementation | Optimized Pipeline |
|-----------------------------------|----------------------|--------------------|
| Single API Task Creation          | 30–45 s              | 1.2–1.8 s          |
| Visual Search Query Response      | 35–50 s              | 2.0–3.5 s          |
| Sequential Batch (5 tasks)        | 175–225 s            | 8.5–10.0 s         |
| Parallel Batch (3 concurrent)     | 60–75 s              | 3.0–5.0 s          |

## 2. End-to-End Detection Latency

| Pipeline Stage            | Naive Python Version | Optimized Version |
|---------------------------|----------------------|-------------------|
| Capture-to-Frame Ready    | 0.25 s               | 0.05 s            |
| Object Detection (YOLO)   | 1.5 s/frame          | 0.03 s/frame      |
| Post-Processing & UI Draw | 0.75 s               | 0.02 s            |
| Total per Cycle           | ~2.5 s               | ~0.20 s           |

## 3. Frame Throughput (FPS)

| Implementation      | Resolution | FPS (frames/sec) |
|---------------------|------------|------------------|
| Naive Python Loop   | 1080p      | 0.4              |
| NumPy-Vectorized    | 640p       | 8.0              |
| Full Optimized      | 640p       | 10.1             |

## 4. Arithmetic Hot Loop (1e6 Operations)

| Operation  | Python Loop Time | Inline Assembly Time | Speedup |
|------------|------------------|----------------------|---------|
| Addition   | 0.15 s           | 0.005 s              | 30×     |
| Multiplication | 0.20 s       | 0.008 s              | 25×     |
| Summation (sum_array) | 0.12 s| 0.006 s              | 20×     |

## 5. Disk & I/O Benchmarks

| Task                          | Naive (PNG per frame) | VideoWriter (MP4) |
|-------------------------------|-----------------------|-------------------|
| Write 100-frame segment       | 18.0 s                | 0.3 s             |
| Read + Downsample 100 frames  | 6.0 s                 | 0.8 s             |

## 6. Resource Utilization

| Metric               | Naive Implementation | Optimized Pipeline |
|----------------------|----------------------|--------------------|
| CPU Usage           | ~90%                 | ~30%               |
| Peak Memory         | ~3.0 GB              | ~1.2 GB            |
| Disk I/O Throughput | ~150 MB/s            | ~50 MB/s           |

---

*Benchmarks were measured on a workstation with a 6‑core Intel i7 CPU, 16 GB RAM, and an NVIDIA GTX GPU. Actual results may vary by hardware.*