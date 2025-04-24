from __future__ import annotations
import cv2, time, sqlite3, tempfile, os, functools, threading, hashlib, datetime, collections
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Deque, List, Tuple

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from twelvelabs import TwelveLabs
from twelvelabs.models.task import Task
from twelvelabs.exceptions import BadRequestError, RateLimitError
from ultralytics import YOLO


@dataclass(slots=True)
class Config:
    api_key   : str = "tlk_1G36X5Q1KS4J5B26BPP8H2WJ2BHR"
    index_id  : str = "6808c0d802327bef162a43b8"
    videos    : dict[str,str] = field(default_factory=lambda: {
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
    tmp_dir    : Path = Path(tempfile.gettempdir()) / "tl_segments"
    videos_path: Path = Path(__file__).parent / "videos"
    db_path    : Path = Path(__file__).with_suffix(".events.db")


CFG = Config()
client = TwelveLabs(api_key=CFG.api_key)



_db = sqlite3.connect(CFG.db_path, check_same_thread=False)
_db.execute("PRAGMA journal_mode=WAL")
_db.execute("""CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, ts TEXT, label TEXT,
  score REAL, confidence REAL, start REAL, end REAL)"""); _db.commit()
def record_event(lbl, sc, cf, stt, end):
    _db.execute("INSERT INTO events VALUES(NULL,datetime('now'),?,?,?,?,?)",
                (lbl, sc, cf, stt, end)); _db.commit()

@functools.lru_cache(1)
def load_yolo(): return YOLO("yolov8n.pt")

def detect_person(frame: np.ndarray
                  ) -> list[Tuple[int,int,int,int]]:
    model = load_yolo()
    res = model.predict(frame, conf=0.5, classes=[0], verbose=False) 
    boxes=[]
    for r in res:
        for *box, conf, cls in r.boxes.data.tolist():
            x1,y1,x2,y2 = map(int, box)
            boxes.append((x1,y1,x2,y2))
    return boxes

class RateLimiter:
    DAILY_LIMIT=45; MIN_INTERVAL=90
    def __init__(self): self.tokens=self.DAILY_LIMIT; self.last_day=datetime.date.today(); self.last_time=0.0
    def allow(self):
        if datetime.date.today()!=self.last_day: self.tokens=self.DAILY_LIMIT; self.last_day=datetime.date.today()
        if self.tokens<=0 or time.time()-self.last_time<self.MIN_INTERVAL: return False
        self.tokens-=1; self.last_time=time.time(); return True
RL=RateLimiter()
last_hash: collections.deque[str]=collections.deque(maxlen=20)
def clip_changed(frames, mask):
    h = hashlib.blake2s()
    for f in frames:
        if mask is not None:
            m = mask
            if m.dtype != np.uint8:
                m = m.astype(np.uint8)
            if m.shape[:2] != f.shape[:2]:
                m = cv2.resize(m, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
            roi = cv2.bitwise_and(f, f, mask=m)
        else:
            roi = f
        h.update(roi)
    d = h.hexdigest()
    if d in last_hash:
        return False
    last_hash.append(d)
    return True

class CapturePipeline:
    def __init__(self, src, fps, seg_len, out_w=640):
        self.cap=cv2.VideoCapture(src)
        self.fps=int(self.cap.get(cv2.CAP_PROP_FPS) or fps)
        self.seg_len=seg_len; self.out_w=out_w; self.dt=1/self.fps
        self.ring:Deque[np.ndarray]=deque(maxlen=self.fps*seg_len*2)
        self.running=False
    def __enter__(self):
        self.running=True; threading.Thread(target=self._loop,daemon=True).start(); return self
    def __exit__(self,*_): self.running=False; self.cap.release()
    def _loop(self):
        while self.running and self.cap.isOpened():
            ok,f=self.cap.read(); 
            if not ok:break
            self.ring.append(f); time.sleep(self.dt)
    def latest(self): return self.ring[-1] if self.ring else None
    def _down(self,f):
        if self.out_w is None or f.shape[1]<=self.out_w: return f
        h,w=f.shape[:2]
        return cv2.resize(f,(self.out_w,int(h*self.out_w/w)),cv2.INTER_AREA)
    def segment(self, roi_mask):
        need = self.fps * self.seg_len
        if len(self.ring) < need:
            return None
        frames = list(self.ring)[-need:]
        if len(frames) < self.fps * 4:
            return None
        frames_ds = [self._down(f) for f in frames]
        if not clip_changed(frames_ds, roi_mask):
            return None
        h, w = frames_ds[0].shape[:2]
        p = CFG.tmp_dir / f"seg_{time.time_ns()}.mp4"
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for f in frames_ds:
            vw.write(f)
        vw.release()
        return p


COMPOSITE_QUERIES: dict[str, list[str]] = {
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

EXEC=ThreadPoolExecutor(max_workers=CFG.max_workers)
class TLJob:
    def __init__(self, clip, query, index_id, parent_label):
        self.clip = clip
        self.query = query
        self.index_id = index_id
        self.parent_label = parent_label 
    def start(self): self.fut=EXEC.submit(self._run); return self
    def _run(self):
        try:
            task = client.task.create(index_id=self.index_id, file=str(self.clip))
            task.wait_for_done(sleep_interval=0.7)
            if task.status != "ready":
                return []
            res = client.search.query(index_id=self.index_id, query_text=self.query, options=["visual"], group_by="video")
            vid = task.video_id
            hits = []
            for g in res.data.root:
                for c in g.clips.root:
                    if c.video_id != vid:
                        continue
                    try:
                        sc = float(c.score)
                        cf = float(c.confidence)
                    except (ValueError, TypeError):
                        continue
                    hits.append((c.start, c.end, sc, cf))
            print(f"Raw TL hits for “{self.query}”: ", hits)
            score_th, conf_th = 0.5, 0.5
            print(f"Applying TL thresholds: score ≥ {score_th}, confidence ≥ {conf_th}")
            hits = [h for h in hits if h[2] >= score_th and h[3] >= conf_th]
            if hits:
                print(f"✅ TL match for “{self.query}” →", hits)
            else:
                print(f"❌ No TL match for “{self.query}”")
            return hits
        except (BadRequestError, RateLimitError) as e:
            print("⚠️ TL:", e)
            return []
        finally:
            self.clip.unlink(missing_ok=True)

def make_mask(frame, pts):
    m=np.zeros(frame.shape[:2],np.uint8)
    if len(pts)>=3: cv2.fillPoly(m,[pts],1)
    return m

def main():
    st.set_page_config("Security Demo","🛡️",layout="wide")
    st.title("⚡ Real-Time Security Detection (w/ YOLO boxes)")

    with st.sidebar:
        current_index_id = CFG.index_id 

        src_sel  = st.selectbox("Source", ["Webcam"] + list(CFG.videos))
        ev_sel =st.selectbox("Event",list(CFG.queries))
        fps    =st.slider("FPS",5,30,CFG.grab_fps)
        seg_len=st.slider("Segment length (s, ≥4)",4,10,CFG.segment_sec)
        st.write("Draw restricted rectangle")
        cvs=st_canvas(fill_color="rgba(255,0,0,0.3)",stroke_color="#ff0000",
                      stroke_width=2,background_color="#00000000",
                      width=400,height=300,drawing_mode="rect",key="zone")
        run=st.toggle("▶️ Run",value=False)

    if cvs.json_data and cvs.json_data.get("objects"):
        o=cvs.json_data["objects"][0]; L,T=o["left"],o["top"]; W,H=o["width"],o["height"]
        pts=np.array([[L,T],[L+W,T],[L+W,T+H],[L,T+H]],int)
    else: pts=np.empty((0,2),int)

    if not run: st.info("Toggle **Run** to start."); return

    src_val=0 if src_sel=="Webcam" else str(CFG.videos_path/src_sel)
    mask=None

    with CapturePipeline(src_val,fps,seg_len) as pipe:
        last_query_ts = 0.0  
        jobs:List[TLJob]=[]
        ph_v,ph_e,ph_s=st.empty(),st.empty(),st.empty()
        while run:
            frame=pipe.latest()
            if frame is None: continue
            active_label = ev_sel

            if mask is None and len(pts) >= 3:
                h, w = frame.shape[:2]
                sx, sy = w / 400, h / 300
                scaled_pts = (pts * np.array([sx, sy])).astype(int)
                mask = make_mask(frame, scaled_pts)

            disp = frame.copy()
            if mask is not None:
                overlay = disp.copy()
                overlay[mask.astype(bool)] = (0, 0, 255)        
                alpha = 0.3                                      
                disp = cv2.addWeighted(overlay, alpha, disp, 1 - alpha, 0)

            boxes = detect_person(frame)
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

            ph_v.image(disp, channels="BGR")

            inside = False
            if mask is not None and len(mask):
                for x1,y1,x2,y2 in boxes:
                    cx,cy = (x1+x2)//2, (y1+y2)//2
                    if mask[cy, cx]:
                        inside = True
                        break
            if inside:
                weapon_query = "person holding gun OR knife OR weapon"
                tmp_clip = pipe.segment(mask if mask is not None else np.zeros(frame.shape[:2], np.uint8))
                if tmp_clip and RL.allow():
                    jobs.append(
                        TLJob(tmp_clip, weapon_query, current_index_id, "Weapon inside R‑zone").start()
                    )

            now = time.time()
            if now - last_query_ts >= 10:
                clip = pipe.segment(None)
                if clip and RL.allow():
                    if active_label in COMPOSITE_QUERIES:
                        for sub_q in COMPOSITE_QUERIES[active_label]:
                            jobs.append(TLJob(clip, sub_q, current_index_id, active_label).start())
                    elif active_label in CFG.queries:
                        jobs.append(
                            TLJob(clip, CFG.queries[active_label], current_index_id, active_label).start()
                        )
                    last_query_ts = now

            done,_=wait([j.fut for j in jobs],timeout=0,return_when=FIRST_COMPLETED)
            for fut in done:
                job = next(j for j in jobs if j.fut is fut); jobs.remove(job)
                hits = fut.result()
                if hits:
                    for s, e, sc, cf in hits:
                        record_event(job.query, sc, cf, s, e)
                        success_msg = f"{job.parent_label}: ✅ {job.query} ({sc:.2f})"
                        ph_e.success(success_msg)
                        st.toast(success_msg, icon="✅")
                else:
                    fail_msg = f"{job.parent_label}: ❌ {job.query}"
                    ph_e.error(fail_msg)
                    st.toast(fail_msg, icon="❌")

            ph_s.text(f"Active jobs: {len(jobs)} | tokens left today: {RL.tokens}")

if __name__=="__main__":
    main()
