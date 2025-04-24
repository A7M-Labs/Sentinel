import cv2, time, threading
from collections import deque
from typing import Deque
import numpy as np
from pathlib import Path

class CapturePipeline:
    def __init__(self, src, fps, seg_len, out_w=640):
        self.cap = cv2.VideoCapture(src)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS) or fps)
        self.seg_len = seg_len
        self.out_w = out_w
        self.dt = 1.0 / self.fps
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.fps * seg_len * 2)
        self.running = False

    def __enter__(self):
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def __exit__(self, *args):
        self.running = False
        self.cap.release()

    def _reader(self):
        while self.running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            self.buffer.append(frame)
            time.sleep(self.dt)

    def latest(self) -> np.ndarray | None:
        return self.buffer[-1] if self.buffer else None

    def segment(self, roi_mask: np.ndarray | None) -> Path | None:
        need = self.fps * self.seg_len
        if len(self.buffer) < need:
            return None
        frames = list(self.buffer)[-need:]
        if len(frames) < self.fps * 4:
            return None
        down = []
        for f in frames:
            if self.out_w and f.shape[1] > self.out_w:
                h, w = f.shape[:2]
                f = cv2.resize(f, (self.out_w, int(h * self.out_w / w)), interpolation=cv2.INTER_AREA)
            down.append(f)
        return self._write_segment(down, roi_mask)

    def _write_segment(self, frames: list[np.ndarray], mask: np.ndarray | None) -> Path | None:
        from core.utils import clip_changed
        from core.config import CFG

        if not clip_changed(frames, mask):
            return None

        h, w = frames[0].shape[:2]
        p = CFG.tmp_dir / f"seg_{time.time_ns()}.mp4"
        vw = cv2.VideoWriter(str(p),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             self.fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        return p
