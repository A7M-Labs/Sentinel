import time, hashlib, collections
from pathlib import Path
import cv2
import numpy as np

_last_hash = collections.deque(maxlen=20)

def clip_changed(frames: list[np.ndarray], mask: np.ndarray | None) -> bool:
    h = hashlib.blake2s()
    for f in frames:
        if mask is not None:
            m = mask.astype(np.uint8)
            if m.shape[:2] != f.shape[:2]:
                m = cv2.resize(m, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
            roi = cv2.bitwise_and(f, f, mask=m)
        else:
            roi = f
        h.update(roi)
    digest = h.hexdigest()
    if digest in _last_hash:
        return False
    _last_hash.append(digest)
    return True


def make_mask(frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
    m = np.zeros(frame.shape[:2], np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m


def try_delete(path: Path, retries: int = 3, delay: float = 0.5):
    for _ in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay)