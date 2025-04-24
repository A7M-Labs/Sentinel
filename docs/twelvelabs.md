# TwelveLabs Integration Guide

This document explains how Sentinel integrates with the TwelveLabs Search API, the trigger conditions for API calls, optimizations applied to the query inputs, and best practices for prompt engineering.

## 1. Integration Overview

Sentinel leverages the TwelveLabs Search API to detect semantic events (e.g. weapons, suspicious behavior) in short video segments. TwelveLabs acts as a visual search engine on top of segment clips created by the CapturePipeline.  

Key components:
- **Client**: Initialized once using `TwelveLabs(api_key=CFG.api_key)`.
- **Jobs**: `TLJob` encapsulates an asynchronous TwelveLabs task and search query.
- **Rate Limiter**: Token-bucket algorithm (`RateLimiter`) to enforce daily and per-call limits.

## 2. Trigger Conditions

1. **Restricted-Zone Breach**  
   - When a person’s bounding-box center (via YOLO) enters the user-drawn ROI mask, Sentinel segments the last N seconds and submits a general “weapons” query:
   ```python
   TLJob(
     clip, 
     CFG.queries["Weapon detected"], 
     CFG.index_id, 
     "Weapon detected"
   ).start()
   ```
2. **Periodic Event Queries**  
   - Every 10 seconds (configurable), segments without ROI are submitted for the user-selected event (e.g. “Fire or smoke”, “Suspicious behavior”).  
   - Composite event types are decomposed into sub-queries:
     ```python
     for sub_q in COMPOSITE_QUERIES[ev_sel]:
         TLJob(clip, sub_q, CFG.index_id, ev_sel).start()
     ```

## 3. Query Optimization

- **Pre-defined Templates**  
  All event labels map to carefully crafted Boolean queries in `core/config.py`:
  ```python
  CFG.queries["Weapon detected"] = (
    "person holding gun OR knife OR weapon OR "
    "firearm visible in hand OR blade brandished"
  )
  ```
- **Composite Queries**  
  Group related sub-queries for “Suspicious behavior” and “Unattended package” to improve recall.
- **Segment Length Tuning**  
  Use 5-second windows by default (`segment_sec = 5`), balancing context richness and API cost.
- **Frame Downsampling**  
  Resize frames to `out_w=640` before segmenting to reduce upload size and speed up API processing.

## 4. Rate Limiting & Batching

- **Daily Quota**: `DAILY_LIMIT = 45` calls per day.
- **Minimum Interval**: `MIN_INTERVAL = 90` seconds between calls.
- **Backoff Strategy**: When rate-limited, the pipeline pauses further API tasks until the next interval.
- **Parallelism**: Up to `max_workers = 3` concurrent jobs to maximize throughput without spiking tokens.

## 5. Error Handling

- **BadRequestError**: Invalid queries are silently dropped; logged via `ph_e.error`.
- **RateLimitError**: On hitting limits, `RL.tokens` is zeroed and no further calls are made until reset at midnight.

## 6. Best Practices

- **Prompt Refinement**  
  Iteratively adjust query strings based on false positives/negatives observed in logs.
- **Ground-Truth Validation**  
  Cross-reference TwelveLabs hits with YOLO detections (`detect_person`) to filter spurious results.
- **Local Caching (Future)**  
  Cache repeated queries for identical segments to reduce API usage.
- **Fallback Strategies**  
  In low-credit scenarios, fall back to on-device VA (e.g. OpenCV template matching) rather than drop events.

## 7. Example Workflow

1. User draws a restricted zone in the Streamlit canvas.
2. A subject crosses into the zone → `clip = pipe.segment(mask)`
3. A `TLJob` with the “Weapon detected” query is enqueued.
4. TwelveLabs processes the clip, returns hits with timestamps.
5. Sentinel records events via `record_event` and displays success/failure to the UI.

---

*For full query definitions, see `core/config.py`. For job lifecycle, see `core/jobs.py`.*  