# Configuration Guide ⚙️

## Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
|`TWELVELABS_API_KEY`|API key for TwelveLabs|Required|
|`INDEX_ID`|Video index identifier|Required|
|`PORT`|Server port number|5001|
|`DEBUG`|Enable debug mode|false|

## Alert Settings
```yaml
alerts:
  confidence_threshold: 0.8
  minimum_clip_duration: 2.0
  notification_channels:
    - email
    - webhook
```

## Performance Tuning
- **Video Processing**
  - Max resolution: 1920x1080
  - Frame rate: 30fps
  - Batch size: 32

- **Memory Usage**
  - Minimum: 4GB RAM
  - Recommended: 8GB RAM