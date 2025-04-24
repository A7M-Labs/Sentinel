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

## Application Settings

| Setting      | Description                                  | Default                               |
|--------------|----------------------------------------------|---------------------------------------|
| segment_sec  | Length (in seconds) of each video segment    | 5                                     |
| grab_fps     | Frames per second captured by the pipeline   | 15                                    |
| max_workers  | Maximum concurrent TwelveLabs API jobs       | 3                                     |
| tmp_dir      | Directory for temporary segment files        | `tmp/segments`                        |
| videos_path  | Directory containing source videos           | `videos/`                             |
| db_path      | File path to the events SQLite database      | `databases/events/events.db`          |

## Predefined Videos

These mappings are defined in `core/config.py` under `CFG.videos`.

| Filename      | TwelveLabs Index ID                     |
|---------------|-----------------------------------------|
| normal1.mp4   | `68088a3c352908d3bc50a428`             |
| normal2.mp4   | `68088a3c352908d3bc50a429`             |
| normal3.mp4   | `68088a3c352908d3bc50a42a`             |
| rob1.mp4      | `68088a3c352908d3bc50a42d`             |
| rob2.mp4      | `68088a3c352908d3bc50a42e`             |
| normal4.mp4   | `6808a21a352908d3bc50a45b`             |
| normal5.mp4   | `6808a26c02327bef162a41a8`             |
| rob3.mp4      | `6808a33702327bef162a41af`             |
| rob4.mp4      | `6808a37d669d2e9f3f513bc5`             |
| rob5.mp4      | `6808a49f669d2e9f3f513bda`             |

## Query Templates

These default queries are defined in `core/config.py` under `CFG.queries`.

| Event Label            | Query String                                                                                     |
|------------------------|--------------------------------------------------------------------------------------------------|
| Restricted-zone breach | `person entering restricted area OR person crossing security line OR intruder climbing fence`   |
| Unattended package     | `bag left alone OR suitcase left unattended OR backpack abandoned OR package or box left on floor` |
| Suspicious behavior    | `person running then leaving quickly OR person stealing item OR person looking around nervously OR person loitering near entrance` |
| Weapon detected        | `person holding gun OR knife OR weapon OR firearm visible in hand OR blade brandished`           |
| Fire or smoke          | `visible flames OR smoke rising OR fire in scene`                                                 |
| Fighting               | `people fighting OR violent altercation OR person punching another`                                |
| Vandalism              | `person spray painting wall OR breaking window OR smashing object`                                |