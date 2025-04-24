from __future__ import annotations
import os
import ctypes, mmap, sys
if sys.platform == "win32":
    kernel32 = ctypes.windll.kernel32
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
        ctypes.c_ulong
    )
def make_asm_func(shellcode: bytes, restype, argtypes):
    size = len(shellcode)
    if sys.platform == "win32":
        MEM_COMMIT  = 0x1000
        MEM_RESERVE = 0x2000
        PAGE_EXECUTE_READWRITE = 0x40
        ptr = kernel32.VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE)
        if not ptr:
            raise MemoryError
        ctypes.memmove(ptr, shellcode, size)
        addr = ptr
    else:
        prot = mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC
        buf = mmap.mmap(-1, size, prot=prot)
        buf.write(shellcode)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

_add_shellcode = bytes([
    0x55,                   # push rbp
    0x48, 0x89, 0xE5,       # mov rbp, rsp
    0x89, 0xF8,             # mov eax, edi
    0x01, 0xF0,             # add eax, esi
    0x5D,                   # pop rbp
    0xC3                    # ret
])
add = make_asm_func(_add_shellcode, ctypes.c_int, (ctypes.c_int, ctypes.c_int))

_sub_shellcode = bytes([
    0x55,                   # push rbp
    0x48, 0x89, 0xE5,       # mov rbp, rsp
    0x89, 0xF8,             # mov eax, edi
    0x29, 0xF0,             # sub eax, esi
    0x5D,                   # pop rbp
    0xC3                    # ret
])
sub = make_asm_func(_sub_shellcode, ctypes.c_int, (ctypes.c_int, ctypes.c_int))

_mul_shellcode = bytes([
    0x55,                   # push rbp
    0x48, 0x89, 0xE5,       # mov rbp, rsp
    0x89, 0xF8,             # mov eax, edi
    0x0F, 0xAF, 0xC6,       # imul eax, esi
    0x5D,                   # pop rbp
    0xC3                    # ret
])
mul = make_asm_func(_mul_shellcode, ctypes.c_int, (ctypes.c_int, ctypes.c_int))

_sum_shellcode = bytes([
    0x48,0x31,0xC0,         # xor    rax, rax
    0x48,0x31,0xC9,         # xor    rcx, rcx
    0x48,0x39,0xCA,         # cmp    rcx, rdx
    0x7D,0x0C,              # jge    +12
    0x8B,0x14,0x0F,         # mov    edx, [rdi + rcx*4]
    0x01,0xD0,              # add    eax, edx
    0x48,0xFF,0xC1,         # inc    rcx
    0xEB,0xF4,              # jmp    -12
    0xC3                    # ret
])
sum_array = make_asm_func(_sum_shellcode, ctypes.c_int, (ctypes.POINTER(ctypes.c_int), ctypes.c_int))

os.environ["STREAMLIT_WATCHER_IGNORE_PATTERNS"] = "torch*"
import cv2, time, sqlite3, tempfile, functools, threading, hashlib, datetime, collections
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Deque, List, Tuple
import numpy as np
import torch
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from twelvelabs import TwelveLabs
from twelvelabs.exceptions import BadRequestError, RateLimitError
from ultralytics import YOLO

@dataclass(slots=True)
class Config:
    api_key    : str = ""
    index_id   : str = ""
    videos     : dict[str,str] = field(default_factory=lambda: {
        "normal1.mp4": "68088a3c352908d3bc50a428",
        "normal2.mp4": "68088a3c352908d3bc50a429",
        "normal3.mp4": "68088a3c352908d3bc50a42a",
        "rob1.mp4"   : "68088a3c352908d3bc50a42d",
        "rob2.mp4"   : "68088a3c352908d3bc50a42e",
        "normal4.mp4":"6808a21a352908d3bc50a45b",
        "normal5.mp4":"6808a26c02327bef162a41a8",
        "rob3.mp4":"6808a33702327bef162a41af",
        "rob4.mp4":"6808a37d669d2e9f3f513bc5",
        "rob5.mp4":"6808a49f669d2e9f3f513bda",
    })
    queries    : dict[str,str] = field(default_factory=lambda:{
        "Restricted-zone breach": (
            "person entering restricted area OR "
            "person crossing security line OR "
            "intruder climbing fence"
        ),
        "Unattended package": (
            "bag left alone OR "
            "suitcase left unattended OR "
            "backpack abandoned OR "
            "package or box left on floor"
        ),
        "Suspicious behavior": (
            "person running then leaving quickly OR "
            "person stealing item OR "
            "person looking around nervously OR "
            "person loitering near entrance"
        ),
        "Weapon detected": (
            "person holding gun OR knife OR weapon OR "
            "firearm visible in hand OR "
            "blade brandished"
        ),
        "Fire or smoke": (
            "visible flames OR "
            "smoke rising OR "
            "fire in scene"
        ),
        "Fighting": (
            "people fighting OR "
            "violent altercation OR "
            "person punching another"
        ),
        "Vandalism": (
            "person spray painting wall OR "
            "breaking window OR "
            "smashing object"
        )
    })
    segment_sec: int = 5
    grab_fps   : int = 15
    max_workers: int = 3
    tmp_dir    : Path = Path(__file__).parent / "tmp" / "segments"
    videos_path: Path = Path(__file__).parent / "videos"
    db_path    : Path = Path(__file__).parent / "databases" / "events" / "events.db"

    def __post_init__(self):
        # create tmp segments folder
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        # ensure the events database folder exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

CFG = Config()
client = TwelveLabs(api_key=CFG.api_key)

_db = sqlite3.connect(CFG.db_path, check_same_thread=False)
_db.execute("PRAGMA journal_mode=WAL")
_db.execute("""
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY,
  ts TEXT,
  label TEXT,
  score REAL,
  confidence REAL,
  start REAL,
  end REAL
)
""")
_db.commit()

def record_event(label: str, score: float, confidence: float, start: float, end: float):
    _db.execute(
        "INSERT INTO events VALUES(NULL, datetime('now'), ?, ?, ?, ?, ?)",
        (label, score, confidence, start, end)
    )
    _db.commit()

class RateLimiter:
    DAILY_LIMIT = 45
    MIN_INTERVAL = 90
    def __init__(self):
        self.tokens = self.DAILY_LIMIT
        self.last_day = datetime.date.today()
        self.last_time = 0.0
    def allow(self) -> bool:
        today = datetime.date.today()
        if today != self.last_day:
            self.tokens = self.DAILY_LIMIT
            self.last_day = today
        now = time.time()
        if self.tokens <= 0 or (now - self.last_time) < self.MIN_INTERVAL:
            return False
        self.tokens -= 1
        self.last_time = now
        return True

RL = RateLimiter()
last_hash: collections.deque[str] = collections.deque(maxlen=20)

def clip_changed(frames: List[np.ndarray], mask: np.ndarray | None) -> bool:
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
    if digest in last_hash:
        return False
    last_hash.append(digest)
    return True

@functools.lru_cache(1)
def load_yolo() -> YOLO:
    # load from models/yolo
    model_path = Path(__file__).parent / "models" / "yolo" / "yolov8n.pt"
    model = YOLO(str(model_path))
    if torch.cuda.is_available():
        model.to("cuda")
    model.fuse()
    return model

def detect_person(frame: np.ndarray) -> List[Tuple[int,int,int,int]]:
    results = load_yolo().predict(
        source=frame,
        conf=0.5,
        classes=[0],
        device=0 if torch.cuda.is_available() else -1,
        verbose=False
    )
    boxes = []
    for r in results:
        for *b, conf, cls in r.boxes.data.tolist():
            x1,y1,x2,y2 = map(int, b)
            boxes.append((x1,y1,x2,y2))
    return boxes

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
                f = cv2.resize(f, (self.out_w, int(h*self.out_w/w)), interpolation=cv2.INTER_AREA)
            down.append(f)
        if not clip_changed(down, roi_mask):
            return None
        h, w = down[0].shape[:2]
        p = CFG.tmp_dir / f"seg_{time.time_ns()}.mp4"
        vw = cv2.VideoWriter(str(p),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             self.fps, (w, h))
        for f in down:
            vw.write(f)
        vw.release()
        return p

COMPOSITE_QUERIES = {
    "Suspicious behavior": [
        "person running then leaving quickly",
        "person holding gun OR knife OR weapon",
        "person stealing item"
    ],
    "Unattended package": [
        "bag left alone",
        "package or box left unattended",
        "unattended luggage"
    ]
}

EXEC = ThreadPoolExecutor(max_workers=CFG.max_workers)

def try_delete(path: Path, retries: int = 3, delay: float = 0.5):
    for _ in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay)

class TLJob:
    def __init__(self, clip: Path, query: str, index_id: str, parent_label: str):
        self.clip = clip
        self.query = query
        self.index_id = index_id
        self.parent_label = parent_label
    def start(self):
        self.fut = EXEC.submit(self._run)
        return self
    def _run(self):
        try:
            task = client.task.create(index_id=self.index_id, file=str(self.clip))
            task.wait_for_done(sleep_interval=0.7)
            if task.status != "ready":
                return []
            res = client.search.query(
                index_id=self.index_id,
                query_text=self.query,
                options=["visual"],
                group_by="video"
            )
            vid = task.video_id
            hits = []
            for g in res.data.root:
                for c in g.clips.root:
                    if c.video_id != vid:
                        continue
                    sc, cf = float(c.score), float(c.confidence)
                    hits.append((c.start, c.end, sc, cf))
            return [h for h in hits if h[2] >= 0.5 and h[3] >= 0.5]
        except RateLimitError:
            RL.tokens = 0
            return []
        except BadRequestError:
            return []
        finally:
            try_delete(self.clip)

def make_mask(frame: np.ndarray, pts: np.ndarray) -> np.ndarray:
    m = np.zeros(frame.shape[:2], np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m

def main():
    st.set_page_config("Security Demo", "🛡️", layout="wide")
    st.title("⚡ Real-Time Security Detection (w/ YOLO boxes)")
    with st.sidebar:
        current_index_id = CFG.index_id
        st.write("add(3,4) =", add(3,4))
        src_sel  = st.selectbox("Source", ["Webcam"] + list(CFG.videos))
        ev_sel   = st.selectbox("Event", list(CFG.queries))
        fps      = st.slider("FPS", 5, 30, CFG.grab_fps)
        seg_len  = st.slider("Segment length", 4, 10, CFG.segment_sec)
        cvs = st_canvas(fill_color="rgba(255,0,0,0.3)",
                        stroke_color="#ff0000", stroke_width=2,
                        background_color="#00000000",
                        width=400, height=300, drawing_mode="rect",
                        key="zone")
        run = st.toggle("Run", value=False)
    if cvs.json_data and cvs.json_data.get("objects"):
        o = cvs.json_data["objects"][0]
        L, T = o["left"], o["top"]
        W, H = o["width"], o["height"]
        pts = np.array([[L, T], [L+W, T], [L+W, T+H], [L, T+H]], int)
    else:
        pts = np.empty((0,2), int)
    if not run:
        st.info("Toggle Run to start.")
        return
    src_val = 0 if src_sel=="Webcam" else str(CFG.videos_path/src_sel)
    mask = None
    with CapturePipeline(src_val, fps, seg_len) as pipe:
        last_query_ts = 0.0
        jobs = []
        ph_v, ph_e, ph_s = st.empty(), st.empty(), st.empty()
        while run:
            frame = pipe.latest()
            if frame is None:
                continue
            if mask is None and len(pts)==4:
                h, w = frame.shape[:2]
                scaled = (pts * np.array([w/400, h/300])).astype(int)
                mask = make_mask(frame, scaled)
            disp = frame.copy()
            if mask is not None:
                overlay = disp.copy()
                overlay[mask.astype(bool)] = (0,0,255)
                disp = cv2.addWeighted(overlay,0.3,disp,0.7,0)
            boxes = detect_person(frame)
            for x1,y1,x2,y2 in boxes:
                cx = add(x1,x2)//2
                cy = add(y1,y2)//2
                w_box = sub(x2,x1)
                h_box = sub(y2,y1)
                area = mul(w_box,h_box)
                cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(disp,(cx,cy),3,(255,0,0),-1)
                cv2.putText(disp,f"{area}",(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            ph_v.image(disp,channels="BGR")
            inside = False
            if mask is not None:
                for x1,y1,x2,y2 in boxes:
                    cx = add(x1,x2)//2
                    cy = add(y1,y2)//2
                    if mask[cy,cx]:
                        inside = True
                        break
            if inside:
                clip = pipe.segment(mask)
                if clip and RL.allow():
                    jobs.append(TLJob(clip,CFG.queries["Weapon detected"],
                                      CFG.index_id,"Weapon detected").start())
            now = time.time()
            if now-last_query_ts>=10:
                clip = pipe.segment(None)
                if clip and RL.allow():
                    if ev_sel in COMPOSITE_QUERIES:
                        for sub_q in COMPOSITE_QUERIES[ev_sel]:
                            jobs.append(TLJob(clip,sub_q,CFG.index_id,ev_sel).start())
                    else:
                        jobs.append(TLJob(clip,CFG.queries[ev_sel],CFG.index_id,ev_sel).start())
                    last_query_ts = now
            done,_ = wait([j.fut for j in jobs],timeout=0,return_when=FIRST_COMPLETED)
            for fut in done:
                job = next(j for j in jobs if j.fut is fut)
                jobs.remove(job)
                hits = fut.result()
                if hits:
                    for s,e,sc,cf in hits:
                        record_event(job.query,sc,cf,s,e)
                        msg = f"{job.parent_label}: ✅ {job.query} ({sc:.2f})"
                        ph_e.success(msg)
                        st.toast(msg,icon="✅")
                else:
                    msg = f"{job.parent_label}: ❌ {job.query}"
                    ph_e.error(msg)
                    st.toast(msg,icon="❌")
            ph_s.text(f"Active jobs: {len(jobs)} | tokens left: {RL.tokens}")

if __name__=="__main__":
    main()
