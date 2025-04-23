import os, sys, time, threading
from collections import deque
from pathlib import Path

import cv2, numpy as np
import torch
from PIL import Image
import streamlit as st
st.set_page_config(page_title="Smart Monitor", layout="wide", page_icon="📹")

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from transformers import CLIPProcessor, CLIPModel

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
FLOAT16 = device.type in ("mps","cuda")

def cast(t):
    return t.to(device, dtype=torch.float16 if FLOAT16 else torch.float32)

VIDEO_DIR = Path("./videos")
YOLO_WEIGHTS = "yolov8n.pt"
LABELS = [
    "person", "man", "woman", "child", "baby", "teenager", "adult", "elderly person",
    "group of people", "couple", "family", "crowd", "solo person",
    "person wearing red shirt", "person wearing blue shirt", "person wearing green shirt",
    "person wearing black hoodie", "person wearing gray hoodie", "person wearing white T-shirt",
    "person wearing dress", "person wearing skirt", "person wearing shorts", "person wearing jeans",
    "person wearing suit", "person wearing tie", "person wearing blouse", "person wearing jacket",
    "person wearing coat", "person wearing raincoat", "person wearing scarf", "person wearing hat",
    "person wearing cap", "person wearing beanie", "person wearing helmet", "person wearing sunglasses",
    "person wearing mask", "person wearing gloves", "person wearing boots", "person wearing sandals",
    "person wearing sneakers", "person wearing high heels", "person wearing backpack",
    "person with umbrella", "person with suitcase", "person with shopping bag",
    "person with briefcase", "person with handbag", "person with laptop bag",
    "person with helmet", "person with bicycle helmet", "person with microphone",
    "person with camera", "person with phone", "person with tablet", "person with book",
    "person standing", "person sitting", "person crouching", "person kneeling",
    "person lying down", "person bending over", "person leaning", "person stretching",
    "person waving", "person with arms crossed",
    "person walking", "person running", "person jogging", "person sprinting",
    "person jumping", "person dancing", "person skipping", "person hopping",
    "person climbing", "person descending stairs", "person riding bicycle",
    "person riding motorcycle", "person in wheelchair", "person pushing stroller",
    "person pushing wheelchair", "person pushing cart", "person pushing wheelchair",
    "person pulling cart", "person holding phone", "person texting", "person calling on phone",
    "person taking photo", "person recording video", "person waving hand",
    "person pointing", "person clapping", "person giving thumbs up",
    "person giving thumbs down", "person shaking hands", "person folding arms",
    "person raising hand", "person carrying child", "person holding baby",
    "person feeding baby", "person holding beverage", "person holding camera",
    "person carrying backpack", "person holding umbrella open", "person holding umbrella closed",
    "person smiling", "person frowning", "person laughing", "person crying",
    "person yawning", "person blowing kiss", "person wearing glasses", "person wearing sunglasses",
    "person wearing face mask",
    "people talking", "people hugging", "people fighting", "people kissing",
    "people dancing together", "group meeting", "group discussion", "person teaching",
    "person listening", "person reading newspaper", "person reading book",
    "person writing", "person drawing", "person painting", "person cooking",
    "person eating", "person drinking", "person smoking",
    "unattended bag", "left luggage", "package", "knife", "gun", "crowbar",
    "vandalism", "graffiti", "trespassing", "tailgating", "jaywalking",
    "person loitering", "suspicious behavior", "person peeking", "person hiding",
    "person breaking window", "person picking lock", "alarm triggered",
    "car", "truck", "bus", "motorcycle", "bicycle", "scooter", "van",
    "train", "tram", "subway", "boat", "airplane", "helicopter",
    "person getting into car", "person getting out of car", "person loading trunk",
    "person unloading trunk", "person boarding bus", "person exiting bus",
    "indoor scene", "outdoor scene", "street scene", "parking lot", "sidewalk",
    "crosswalk", "staircase", "elevator", "building interior", "office environment",
    "warehouse", "park", "playground", "shopping mall", "restaurant", "cafe",
    "beach", "mountain", "forest", "highway", "train station", "airport",
    "laptop", "tablet", "smartphone", "book", "newspaper", "magazine",
    "backpack", "handbag", "suitcase", "shopping cart", "trolley", "wheelchair",
    "stroller", "umbrella", "microphone", "camera", "bottle", "cup",
    "box", "package", "briefcase", "toolbox", "ladder", "crate",
    "person shopping", "person waiting", "person relaxing", "person meditating",
    "person exercising", "person lifting weights", "person doing yoga",
    "person playing guitar", "person playing piano", "person singing",
    "person speaking", "person presenting", "person teaching class",
    "person studying", "person working on computer", "person coding",
    "shadow", "reflection", "silhouette", "motion blur", "low light",
    "night scene", "daytime scene", "rainy weather", "snowy weather",
    "sunny day", "foggy weather", "construction site", "roadwork",
    "emergency vehicle", "ambulance", "fire truck", "police car",
    "handgun", "revolver", "pistol", "assault rifle", "rifle", "shotgun",
    "submachine gun", "sniper rifle", "machine gun", "grenade", "explosive device",
    "person holding handgun", "person holding rifle", "person holding shotgun",
    "person firing weapon", "gunshot fired", "person pointing gun at person",
    "person with knife drawn", "person holding crowbar",
    "armed robbery", "bank robbery", "shoplifting", "pickpocketing", "mugging",
    "burglary", "breaking into car", "breaking into building", "extortion",
    "threatening with weapon", "demanding money", "stealing bag", "stealing wallet",
    "person pickpocketing", "person shoplifting", "person committing robbery",
    "person looting", "person vandalizing property", "person breaking lock"
]
TOP_K = 3
YOLO_CONF = 0.25
ENTRANCE = [(0,360),(300,360),(300,480),(0,480)]

@st.cache_resource(show_spinner=True)
def load_detector():
    yolo = YOLO(YOLO_WEIGHTS)
    yolo.fuse()
    yolo.model.model = torch.quantization.quantize_dynamic(
        yolo.model.model,
        {torch.nn.Conv2d},
        dtype=torch.qint8
    )
    yolo.args.imgsz = 320
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = clip_proc.tokenizer
    txt = tokenizer(LABELS, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**txt)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return yolo, clip_model, clip_proc, text_emb

yolo, clip_model, clip_proc, TEXT_EMBEDS = load_detector()
tracker = DeepSort(max_age=30)

class AVThread:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path), cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.q = deque(maxlen=120)
        self.stop = False
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while not self.stop:
            ret, frm = self.cap.read()
            if not ret:
                self.stop = True
                break
            self.q.append(frm)
    def read(self):
        return self.q.popleft() if self.q else None
    def release(self):
        self.stop = True
        self.cap.release()

@torch.no_grad()
def classify(crop_bgr):
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    inputs = clip_proc(images=img, return_tensors="pt", padding=True).to(device)
    img_emb = clip_model.get_image_features(**inputs)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    sims = (img_emb @ TEXT_EMBEDS.T).squeeze(0)
    top = sims.topk(TOP_K)
    return [(LABELS[i], sims[i].item()) for i in top.indices]

st.title("📹 Smart Video Monitor – Apple Silicon")

vids = [p.name for p in VIDEO_DIR.glob("*.mp4")]
if not vids:
    st.error("Put MP4s in ./videos")
    st.stop()
name = st.selectbox("Video", vids)
if not st.button("▶️ Run"):
    st.stop()

cap = AVThread(VIDEO_DIR/name)
poly = np.array(ENTRANCE)
frame_ph, occ_ph = st.empty(), st.empty()
class_box = st.container()
occ = 0
prev_in = {}

while True:
    frame = cap.read()
    if frame is None:
        if cap.stop:
            break
        time.sleep(0.01)
        continue
    if int(cap.cap.get(cv2.CAP_PROP_POS_FRAMES)) % 3 == 0:
        detections = []
        result = yolo(frame, classes=[0], conf=YOLO_CONF, verbose=False)[0]
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            labels = classify(crop)
            detections.append(((x1, y1, x2, y2), conf.item(), labels[0]))
        track_in = [[[x1, y1, x2-x1, y2-y1], c, 0] for (x1, y1, x2, y2), c, _ in detections]
        tracks = tracker.update_tracks(track_in, frame=frame)
    else:
        detections = []
        tracks = []
    for det, t in zip(detections, tracks):
        if not t.is_confirmed():
            continue
        (x1, y1, x2, y2), conf, (lbl, sc) = det
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        inside = cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
        if prev_in.get(t.track_id, False) != inside:
            occ += 1 if inside else -1
            occ = max(0, occ)
        prev_in[t.track_id] = inside
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{lbl} {sc:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.polylines(frame, [poly], True, (255,0,0), 2)
    cv2.putText(frame, f"Occupancy: {occ}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    occ_ph.metric("Occupancy", occ)
    with class_box:
        st.markdown("### Recent labels")
        for _, _, (lbl, sc) in detections[:10]:
            st.write(f"{lbl} ({sc:.2f})")
cap.release()
st.success("Done ✔️")