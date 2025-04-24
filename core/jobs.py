import time, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from twelvelabs import TwelveLabs
from twelvelabs.exceptions import BadRequestError, RateLimitError

from core.config import CFG
from core.utils import try_delete

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
EXEC = ThreadPoolExecutor(max_workers=CFG.max_workers)

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

client = TwelveLabs(api_key=CFG.api_key)

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