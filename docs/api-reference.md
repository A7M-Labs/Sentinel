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