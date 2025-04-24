# System Limitations & Challenges 🚧

## API Constraints

### Rate Limiting
- **Issue**: Frequent API rate limiting encountered during high-volume processing  
- **Impact**: Causes interruptions in real-time monitoring  
- **Workaround**: Implemented request queuing and exponential backoff strategy  
  ```python
  MAX_RETRIES = 3
  BACKOFF_FACTOR = 2
  INITIAL_BACKOFF = 30  # seconds
  ```

### API Credit Usage
- Daily credit limits affect continuous monitoring capabilities  
- Processing high-resolution videos consumes credits rapidly  
- Current credit allocation allows approximately:
  - 2 hours of continuous monitoring
  - 10 video file analyses per day

## Performance Bottlenecks

### Processing Latency
- Average processing time: 2–3 seconds per frame  
- Batch processing improves throughput but increases end-to-end latency  
- Network latency adds 0.5–1.5 seconds to response time  

### Resource Usage for Video Processing (Not Search API)
| Resource | Limitation                                    |
|----------|-----------------------------------------------|
| CPU      | Peaks at 80% under 1080p, 30 FPS workloads    |
| Memory   | Up to 1.2 GB per segment buffer              |
| Disk I/O | Temporary segment writes can saturate SSD     |
| Time     | Minimum 48.4 s for full-segment write/read   |

## Code-Level Limitations

1. **Executable Memory Allocation**  
   - Shellcode functions use `mmap`/`VirtualAlloc` for RWX pages  
   - **Risk**: Platform compatibility issues and potential security flags (DEP, W^X)  
   - **Impact**: May fail or be blocked on hardened systems

2. **Thread Safety & Concurrency**  
   - SQLite writes (`record_event`) are not fully synchronized across threads  
   - **Risk**: Potential database locks or race conditions under heavy load  
   - **Workaround**: Using WAL mode, but may still see transient write failures

3. **Buffer Management**  
   - Frame buffer capped at `fps * seg_len * 2`  
   - **Issue**: If reader thread lags, buffer underflows or overwrites occur  
   - **Impact**: Missed frames or stale data segments

4. **Temporary File Cleanup**  
   - `tmp/segments/` may accumulate if `try_delete` fails repeatedly  
   - **Risk**: Disk space exhaustion over long runtimes  
   - **Mitigation**: Periodic cleanup script recommended

5. **Hard-Coded Paths & Constants**  
   - Paths for models (`models/yolo/yolov8n.pt`) and DBs are static  
   - **Limitation**: Less flexibility for dynamic deployments or containerization  
   - **Improvement**: Parameterize via environment or CLI flags

6. **Detection Thresholds**  
   - Fixed YOLO confidence threshold (`0.5`) and composite query logic  
   - **Issue**: May not adapt to varying scene conditions  
   - **Future**: Expose thresholds as tunable parameters in UI

## Known Issues

1. **False Positives**  
   - Current confidence threshold: 0.85  
   - False positive rate: ~15%  
   - Working on improved filtering algorithms

2. **Video Quality Requirements**  
   - Minimum resolution: 480p  
   - Maximum supported: 1080p  
   - Frame rate: 15–30 FPS

## Future Improvements

### Planned Enhancements
- [ ] Implement local frame caching  
- [ ] Add request rate–limiting queue  
- [ ] Optimize frame preprocessing pipeline  
- [ ] Add fallback detection methods  

### Alternative Approaches
- Consider hybrid local/cloud processing  
- Implement frame skipping during high load  
- Use adaptive quality scaling  
- Containerize for consistent environment and path management  
