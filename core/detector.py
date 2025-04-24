from functools import lru_cache
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path

@lru_cache(1)
def load_yolo() -> YOLO:
    model_path = Path(__file__).parent.parent / "models" / "yolo" / "yolov8n.pt"
    model = YOLO(str(model_path))
    if torch.cuda.is_available():
        model.to("cuda")
    model.fuse()
    return model


def detect_person(frame: np.ndarray) -> list[tuple[int,int,int,int]]:
    results = load_yolo().predict(
        source=frame,
        conf=0.5,
        classes=[0],
        device=0 if torch.cuda.is_available() else -1,
        verbose=False
    )
    boxes: list[tuple[int,int,int,int]] = []
    for r in results:
        for *b, conf, cls in r.boxes.data.tolist():
            x1,y1,x2,y2 = map(int, b)
            boxes.append((x1,y1,x2,y2))
    return boxes