

# Known Issues & Caveats

While the Sentinel pipeline delivers real-time security detection, several limitations and corner cases remain:

## 1. Limited TwelveLabs API Access
- **API Quota Constraints**  
  Due to restricted daily credits and rate limits, we could only run a limited number of prompt variations and queries.
- **Prompt Tuning**  
  Insufficient API access prevented exhaustive prompt engineering for all event types, leading to some suboptimal query formulations.
- **Impact**  
  Occasional missed detections or lower relevance scores for rare or ambiguous events.

## 2. Edge Case Testing
- **Scenario Coverage**  
  Edge cases (occluded objects, extreme lighting, rapid motion) were tested on a small subset of clips.
- **Impact**  
  Reduced detection accuracy under challenging conditions; may require additional training data or manual annotation.

## 3. Accuracy & Validation
- **YOLO Verification**  
  All TwelveLabs search hits are cross-checked against local YOLO detections to filter out false positives.
- **Residual False Positives/Negatives**  
  Despite dual verification, some events may be misclassified or missed entirely, especially for small objects or partial occlusions.

## 4. Performance Trade‑offs
- Occasional delays when rate limits throttle API calls (handled via backoff but adds latency).
- Temporary segment files in `tmp/segments/` may accumulate if cleanup errors occur.

## 5. Future Mitigations
- Increase TwelveLabs credit allocation or switch to a local embedding-based search model.
- Expand prompt engineering and edge-case datasets.
- Introduce active learning to refine queries and adapt thresholds dynamically.