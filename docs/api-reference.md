# API Reference 📚

## Video Processing API

### Process Video
```http
POST /api/process
Content-Type: multipart/form-data

{
  "video": File,
  "options": {
    "detect_objects": boolean,
    "detect_actions": boolean
  }
}
```

### Search Events
```http
GET /api/search
Query Parameters:
  - query: string
  - threshold: float (0-1)
  - limit: integer
```

## WebSocket Events
| Event | Description |
|-------|-------------|
|`detection_start`|Video processing initiated|
|`detection_complete`|Processing completed|
|`alert`|Security event detected|

## Events API

### List Events
```http
GET /api/events
Content-Type: application/json
Query Parameters:
  - start_ts: string (ISO 8601, optional)
  - end_ts: string (ISO 8601, optional)
  - label: string (optional)
  - limit: integer (optional, default=100)
```

**Response**
```json
[
  {
    "id": 123,
    "ts": "2025-04-23T10:15:30Z",
    "label": "Weapon detected",
    "score": 0.92,
    "confidence": 0.87,
    "start": 12.5,
    "end": 17.5
  },
  ...
]
```

### Get Event
```http
GET /api/events/{id}
```
**Response**
```json
{
  "id": 123,
  "ts": "2025-04-23T10:15:30Z",
  "label": "Weapon detected",
  "score": 0.92,
  "confidence": 0.87,
  "start": 12.5,
  "end": 17.5
}
```

### Delete Event
```http
DELETE /api/events/{id}
```
**Response**
204 No Content

## Configuration API

### Get Configuration
```http
GET /api/config
```
**Response**
```json
{
  "segment_sec": 5,
  "grab_fps": 15,
  "max_workers": 3,
  "index_id": "68088a3c352908d3bc50a428"
}
```

### Update Configuration
```http
PUT /api/config
Content-Type: application/json

{
  "segment_sec": 10,
  "grab_fps": 20
}
```
**Response**
```json
{
  "segment_sec": 10,
  "grab_fps": 20,
  "max_workers": 3,
  "index_id": "68088a3c352908d3bc50a428"
}
```

## WebSocket Events (Extended)
| Event               | Payload                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `detection_start`   | `{ "timestamp": "2025-04-23T10:00:00Z", "source": "Webcam" }`           |
| `detection_complete`| `{ "timestamp": "2025-04-23T10:00:05Z", "duration": 5.2 }`             |
| `alert`             | `{ "label": "Weapon detected", "score": 0.92, "confidence": 0.87, "start": 12.5, "end": 17.5, "timestamp": "2025-04-23T10:00:12Z" }` |
| `status_update`     | `{ "active_jobs": 2, "tokens_left": 43 }`                                |