from __future__ import annotations
import os
os.environ["STREAMLIT_WATCHER_IGNORE_PATTERNS"] = "torch*"
import cv2, time
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from concurrent.futures import wait, FIRST_COMPLETED

from core.config import CFG
from core.asm_funcs import add, sub, mul
from core.detector import detect_person
from core.pipeline import CapturePipeline
from core.jobs import TLJob, RL, COMPOSITE_QUERIES
from core.utils import make_mask
from core.db import record_event


def main():
    st.set_page_config("Security Demo", layout="wide")
    st.title("Real-Time Security Detection")
    with st.sidebar:
        st.write("add(3,4) =", add(3,4))
        src_sel = st.selectbox("Source", ["Webcam"] + list(CFG.videos))
        ev_sel = st.selectbox("Event", list(CFG.queries))
        fps = st.slider("FPS", 5, 30, CFG.grab_fps)
        seg_len = st.slider("Segment length", 4, 10, CFG.segment_sec)
        cvs = st_canvas(
            fill_color="rgba(255,0,0,0.3)", stroke_color="#ff0000", stroke_width=2,
            background_color="#00000000", width=400, height=300,
            drawing_mode="rect", key="zone"
        )
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

    src_val = 0 if src_sel == "Webcam" else str(CFG.videos_path / src_sel)
    mask = None

    with CapturePipeline(src_val, fps, seg_len) as pipe:
        last_query_ts = 0.0
        jobs: list[TLJob] = []
        ph_v, ph_e, ph_s = st.empty(), st.empty(), st.empty()
        while run:
            frame = pipe.latest()
            if frame is None:
                continue

            if mask is None and pts.shape[0] == 4:
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
                cx = add(x1, x2)//2
                cy = add(y1, y2)//2
                w_box = sub(x2, x1)
                h_box = sub(y2, y1)
                area = mul(w_box, h_box)
                cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.circle(disp, (cx,cy), 3, (255,0,0), -1)
                cv2.putText(disp, f"{area}", (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            ph_v.image(disp, channels="BGR")

            inside = False
            if mask is not None:
                for x1,y1,x2,y2 in boxes:
                    cx = add(x1, x2)//2
                    cy = add(y1, y2)//2
                    if mask[cy, cx]:
                        inside = True
                        break

            if inside:
                clip = pipe.segment(mask)
                if clip and RL.allow():
                    jobs.append(
                        TLJob(clip, CFG.queries["Weapon detected"], CFG.index_id, "Weapon detected").start()
                    )

            now = time.time()
            if now - last_query_ts >= 10:
                clip = pipe.segment(None)
                if clip and RL.allow():
                    if ev_sel in COMPOSITE_QUERIES:
                        for sub_q in COMPOSITE_QUERIES[ev_sel]:
                            jobs.append(
                                TLJob(clip, sub_q, CFG.index_id, ev_sel).start()
                            )
                    else:
                        jobs.append(
                            TLJob(clip, CFG.queries[ev_sel], CFG.index_id, ev_sel).start()
                        )
                    last_query_ts = now

            done, _ = wait([j.fut for j in jobs], timeout=0, return_when=FIRST_COMPLETED)
            for fut in done:
                job = next(j for j in jobs if j.fut is fut)
                jobs.remove(job)
                hits = fut.result()
                if hits:
                    for s,e,sc,cf in hits:
                        record_event(job.query, sc, cf, s, e)
                        msg = f"{job.parent_label}: ✅ {job.query} ({sc:.2f})"
                        ph_e.success(msg)
                        st.toast(msg, icon="✅")
                else:
                    msg = f"{job.parent_label}: ❌ {job.query}"
                    ph_e.error(msg)
                    st.toast(msg, icon="❌")

            ph_s.text(f"Active jobs: {len(jobs)} | tokens left: {RL.tokens}")


if __name__ == "__main__":
    main()
