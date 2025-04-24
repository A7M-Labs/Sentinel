from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
import tempfile

@dataclass(slots=True)
class Config:
    api_key: str = os.getenv("TWELVELABS_API_KEY", "")
    index_id: str = os.getenv("INDEX_ID", "")
    videos: dict[str,str] = field(default_factory=lambda: {
        "normal1.mp4": "68088a3c352908d3bc50a428",
        "normal2.mp4": "68088a3c352908d3bc50a429",
        "normal3.mp4": "68088a3c352908d3bc50a42a",
        "rob1.mp4":   "68088a3c352908d3bc50a42d",
        "rob2.mp4":   "68088a3c352908d3bc50a42e",
        "normal4.mp4":"6808a21a352908d3bc50a45b",
        "normal5.mp4":"6808a26c02327bef162a41a8",
        "rob3.mp4":   "6808a33702327bef162a41af",
        "rob4.mp4":   "6808a37d669d2e9f3f513bc5",
        "rob5.mp4":   "6808a49f669d2e9f3f513bda",
    })
    queries: dict[str,str] = field(default_factory=lambda: {
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
    grab_fps: int = 15
    max_workers: int = 3
    tmp_dir: Path = Path(__file__).parent.parent / "tmp" / "segments"
    videos_path: Path = Path(__file__).parent.parent / "videos"
    db_path: Path = Path(__file__).parent.parent / "databases" / "events" / "events.db"

    def __post_init__(self):
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

CFG = Config()