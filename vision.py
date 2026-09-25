"""Vision pipelines for the retail prototype.

FloorPipeline  - finds people (YOLO 'person', HOG fallback), tracks them, and measures
                 visitors per section, dwell time, current occupancy and checkout queue.
ShelfPipeline  - watches shelf spots with reference-frame differencing (no ML) and
                 lowers stock by one each time an item is picked up.

Both run in background threads and publish results + annotated frames to the Hub.
All zone rectangles are stored as fractions of the frame: [x, y, w, h] in 0..1.
"""

import math
import os
import threading
import time

import cv2
import numpy as np

from store import now_iso

FONT = cv2.FONT_HERSHEY_SIMPLEX
# BGR colours matching the dashboard palette
TEAL = (139, 111, 31)       # #1F6F8B
MARIGOLD = (41, 180, 240)   # #F0B429
RED = (45, 65, 194)         # #C2412D
WHITE = (240, 242, 240)
INK = (47, 42, 30)          # #1E2A2F


# ------------------------------------------------------------------ helpers
def resolve_source(src, base_dir):
    """Return (source, is_file). Ints / digit strings are cameras."""
    if isinstance(src, int):
        return src, False
    s = str(src).strip()
    if s.isdigit():
        return int(s), False
    if s.startswith(("rtsp://", "http://", "https://")):
        return s, False
    path = s if os.path.isabs(s) else os.path.join(base_dir, s)
    return path, True


def open_capture(src, is_file):
    if is_file and not os.path.exists(src):
        return None, f"Video not found: {os.path.basename(src)}. Put it next to app.py or fix the path in config.json."
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None, ("Video could not be opened." if is_file
                      else f"Camera {src} could not be opened. Try another camera number in config.json.")
    return cap, None


def resize_to(frame, width):
    if width and frame.shape[1] > width:
        h = int(frame.shape[0] * width / frame.shape[1])
        return cv2.resize(frame, (width, h))
    return frame


def rect_px(rect, w, h):
    x, y, rw, rh = rect
    return int(x * w), int(y * h), int(rw * w), int(rh * h)


def in_rect(px, py, r):
    x, y, w, h = r
    return x <= px <= x + w and y <= py <= y + h


def zone_of(px, py, rects):
    for i, r in enumerate(rects):
        if in_rect(px, py, r):
            return i
    return None


def encode(frame, quality=72):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def tag(img, text, x, y, bg, fg=INK, scale=0.5):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    y = max(th + 6, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + base - 1), bg, -1)
    cv2.putText(img, text, (x + 4, y - 3), FONT, scale, fg, 1, cv2.LINE_AA)


# ------------------------------------------------------------------ detection
class PersonDetector:
    """YOLO person detector with an OpenCV HOG fallback (no extra installs)."""

    def __init__(self, cfg):
        self.conf = float(cfg.get("conf", 0.3))
        self.kind = None
        if cfg.get("detector", "yolo") == "test_blobs":
            self.kind = "test_blobs"      # ONLY for the bundled synthetic test video
            return
        if cfg.get("detector", "yolo") == "yolo":
            try:
                from ultralytics import YOLO
                self.model = YOLO(cfg.get("yolo_model", "yolov8s.pt"))
                self.kind = "yolo"
            except Exception as e:  # missing package / no internet for weights
                print(f"[floor] YOLO unavailable ({e}); falling back to OpenCV HOG")
        if self.kind is None:
            self.hog = cv2.HOGDescriptor()
            self.hog.setDefaultPeopleDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.kind = "hog"

    def detect(self, frame):
        """Return a list of (x1, y1, x2, y2) person boxes."""
        if self.kind == "test_blobs":     # white shapes stand in for people
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, m = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return [(x, y, x + w, y + h) for x, y, w, h in map(cv2.boundingRect, cs) if w * h > 500]
        if self.kind == "yolo":
            r = self.model.predict(frame, conf=self.conf, classes=[0], verbose=False)[0]
            return [tuple(map(int, b.xyxy[0])) for b in r.boxes]
        rects, _ = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]


# ------------------------------------------------------------------ tracking
class Tracker:
    """Lightweight nearest-neighbour tracker on foot points (no extra dependencies)."""

    def __init__(self, max_dist_px, max_missed):
        self.max_dist = max_dist_px
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        gone = list(self.tracks.items())
        self.tracks = {}
        return gone

    def update(self, points):
        pairs = []
        for tid, t in self.tracks.items():
            tx, ty = t["pt"]
            for j, (x, y) in enumerate(points):
                d = math.hypot(x - tx, y - ty)
                if d <= self.max_dist:
                    pairs.append((d, tid, j))
        pairs.sort(key=lambda p: p[0])

        used, assigned = set(), {}
        for _, tid, j in pairs:
            if tid in used or j in assigned:
                continue
            used.add(tid)
            assigned[j] = tid
            self.tracks[tid]["pt"] = points[j]
            self.tracks[tid]["missed"] = 0

        for tid, t in self.tracks.items():
            if tid not in used:
                t["missed"] += 1

        for j, p in enumerate(points):
            if j not in assigned:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"pt": p, "missed": 0, "seen": 0, "zone": None,
                                    "cand": None, "cand_n": 0, "enter": 0.0}
                assigned[j] = tid

        expired = [(tid, t) for tid, t in self.tracks.items() if t["missed"] > self.max_missed]
        for tid, _ in expired:
            del self.tracks[tid]
        return assigned, expired


# ------------------------------------------------------------------ floor
class FloorPipeline(threading.Thread):
    def __init__(self, cfg, hub, base_dir):
        super().__init__(daemon=True, name="floor")
        self.fc = cfg["floor"]
        self.tc = cfg.get("tracking", {})
        self.hub = hub
        self.width = cfg.get("process_width", 960)
        self.src, self.is_file = resolve_source(self.fc.get("source", 0), base_dir)
        self.sections = self.fc.get("sections", [])
        q = self.fc.get("queue") or {}
        self.queue = q if q.get("rect") else None
        self.reset_req = threading.Event()
        self.fps = 0.0
        self._last_q_log = 0.0
        self._reset_counters()

    def _reset_counters(self):
        n = len(self.sections)
        self.visits = [0] * n
        self.dwell_total = [0.0] * n
        self.dwell_n = [0] * n
        self.visitors = 0
        self.q_over_since = None
        self.q_alerted = False

    # ---- thread loop ----
    def run(self):
        try:
            self.det = PersonDetector(self.fc)
        except Exception as e:
            self.hub.set_status("floor", "error", f"Person detector failed to load: {e}")
            return
        while True:
            cap, err = open_capture(self.src, self.is_file)
            if cap is None:
                self.hub.set_status("floor", "offline", err)
                time.sleep(3)
                continue
            msg = {"yolo": "YOLO person detection",
                   "hog": "HOG person detection (fallback)",
                   "test_blobs": "Test mode: white shapes stand in for people"}[self.det.kind]
            self.hub.set_status("floor", "live", msg)
            self._stream(cap)
            cap.release()
            if self.is_file and not self.fc.get("loop", True):
                self.hub.set_status("floor", "ended", "Video finished. Set loop to true in config.json to replay.")
                return
            self.hub.set_status("floor", "offline", "Video source stopped. Retrying.")
            time.sleep(2)

    def _stream(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if not (1 <= fps <= 120):
            fps = 25.0
        every = max(1, int(self.fc.get("process_every", 1)))
        realtime = bool(self.fc.get("realtime", True)) and self.is_file
        tracker = None
        clock, t0 = 0.0, time.time()
        idx, since_rewind = 0, 0
        last_proc = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                if self.is_file and self.fc.get("loop", True) and since_rewind > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    since_rewind = 0
                    if tracker:  # people "teleport" on a loop: close their visits cleanly
                        self._finish_tracks(tracker.reset(), clock)
                    continue
                return
            since_rewind += 1
            idx += 1
            clock = clock + 1.0 / fps if self.is_file else time.time() - t0

            if self.reset_req.is_set():
                self.reset_req.clear()
                self._reset_counters()
                if tracker:
                    tracker.reset()
                self.hub.clear_trend()

            if realtime:  # play the video at its natural speed
                lag = clock - (time.time() - t0)
                if lag > 0:
                    time.sleep(min(lag, 0.5))
            if idx % every:
                continue

            frame = resize_to(frame, self.width)
            if tracker is None:
                tracker = Tracker(self.tc.get("max_dist", 0.08) * frame.shape[1],
                                  self.tc.get("max_missed", 12))
            self._process(frame, tracker, clock)

            now = time.time()
            inst = 1.0 / max(1e-3, now - last_proc)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            last_proc = now

    # ---- per frame ----
    def _process(self, frame, tracker, clock):
        h, w = frame.shape[:2]
        secs = [rect_px(s["rect"], w, h) for s in self.sections]
        qrect = rect_px(self.queue["rect"], w, h) if self.queue else None

        boxes = self.det.detect(frame)
        feet = [((b[0] + b[2]) / 2.0, float(b[3])) for b in boxes]  # where each person stands
        assigned, expired = tracker.update(feet)
        self._finish_tracks(expired, clock)

        min_zone = int(self.tc.get("min_zone_frames", 3))
        min_seen = int(self.tc.get("min_track_frames", 3))
        now_counts = [0] * len(secs)
        qlen = 0

        for j, (fx, fy) in enumerate(feet):
            zi = zone_of(fx, fy, secs)
            if zi is not None:
                now_counts[zi] += 1
            if qrect and in_rect(fx, fy, qrect):
                qlen += 1

            tid = assigned[j]
            t = tracker.tracks[tid]
            t["seen"] += 1
            if t["seen"] == min_seen:          # confirmed person, not a flicker
                self.visitors += 1
                self.hub.bump_trend()

            if zi == t["cand"]:
                t["cand_n"] += 1
            else:
                t["cand"], t["cand_n"] = zi, 1
            if t["seen"] >= min_seen and t["cand_n"] >= min_zone and t["zone"] != zi:
                if t["zone"] is not None:
                    self._close_dwell(tid, t, clock)
                t["zone"] = zi
                if zi is not None:
                    t["enter"] = clock
                    self.visits[zi] += 1
                    self.hub.log("INSERT INTO visits VALUES (?,?,?)",
                                 (now_iso(), self.sections[zi]["name"], tid))

        self._check_queue(qlen, clock)
        self._draw(frame, boxes, assigned, secs, qrect, now_counts, qlen)
        self._publish(frame, now_counts, qlen, len(feet))

    def _close_dwell(self, tid, t, clock):
        zi = t["zone"]
        secs = max(0.0, clock - t["enter"])
        self.dwell_total[zi] += secs
        self.dwell_n[zi] += 1
        self.hub.log("INSERT INTO dwell VALUES (?,?,?,?)",
                     (now_iso(), self.sections[zi]["name"], tid, round(secs, 1)))
        t["zone"] = None

    def _finish_tracks(self, items, clock):
        for tid, t in items:
            if t["zone"] is not None and t["zone"] < len(self.sections):
                self._close_dwell(tid, t, clock)

    def _check_queue(self, qlen, clock):
        if not self.queue:
            return
        name = self.queue.get("name", "Checkout")
        limit = int(self.queue.get("alert_at", 4))
        hold = float(self.queue.get("alert_after_s", 5))
        if qlen >= limit:
            if self.q_over_since is None:
                self.q_over_since = clock
            elif not self.q_alerted and clock - self.q_over_since >= hold:
                self.q_alerted = True
                self.hub.alert("long_queue", name,
                               f"{qlen} people waiting at {name}. Open another counter.")
        else:
            self.q_over_since = None
            self.q_alerted = False
        if time.time() - self._last_q_log >= 5:
            self._last_q_log = time.time()
            self.hub.log("INSERT INTO queue_readings VALUES (?,?,?)", (now_iso(), name, qlen))

    def _draw(self, frame, boxes, assigned, secs, qrect, now_counts, qlen):
        top = max(self.visits) if self.visits else 0
        busiest = self.visits.index(top) if top > 0 else -1
        for i, (x, y, rw, rh) in enumerate(secs):
            is_top = i == busiest
            col = MARIGOLD if is_top else TEAL
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), col, 3 if is_top else 2)
            tag(frame, f"{self.sections[i]['name']}  {now_counts[i]} now",
                x + 4, y + 22, col, INK if is_top else WHITE)
        if qrect:
            x, y, rw, rh = qrect
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), RED, 2)
            tag(frame, f"{self.queue.get('name', 'Checkout')} queue: {qlen}",
                x + 4, y + rh - 6, RED, WHITE)
        for j, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 1)
            cv2.circle(frame, (int((x1 + x2) / 2), y2), 4, MARIGOLD, -1)
            tid = assigned.get(j)
            if tid is not None:
                cv2.putText(frame, f"#{tid}", (x1 + 2, y1 + 14), FONT, 0.45, WHITE, 1, cv2.LINE_AA)

    def _publish(self, frame, now_counts, qlen, in_view):
        sections = []
        for i, s in enumerate(self.sections):
            n = self.dwell_n[i]
            sections.append({"name": s["name"], "visits": self.visits[i], "now": now_counts[i],
                             "avg_dwell": round(self.dwell_total[i] / n, 1) if n else None})
        queue = None
        if self.queue:
            queue = {"name": self.queue.get("name", "Checkout"), "length": qlen,
                     "alert_at": int(self.queue.get("alert_at", 4))}
        self.hub.update("floor", {"sections": sections, "visitors": self.visitors,
                                  "in_view": in_view, "queue": queue,
                                  "fps": round(self.fps, 1), "detector": self.det.kind})
        self.hub.set_frame("floor", encode(frame))


# ------------------------------------------------------------------ shelf
class ShelfPipeline(threading.Thread):
    def __init__(self, cfg, hub, base_dir):
        super().__init__(daemon=True, name="shelf")
        self.sc = cfg["shelf"]
        self.hub = hub
        self.width = cfg.get("process_width", 960)
        self.src, self.is_file = resolve_source(self.sc.get("source", 0), base_dir)
        self.lock = threading.Lock()
        self.calibrate_req = threading.Event()
        self.reference = None
        saved = hub.saved_stock()
        self.spots = []
        for s in self.sc.get("spots", []):
            capacity = int(s.get("stock", 10))
            stock = int(saved.get(s["name"], capacity))
            low_at = int(s.get("low_at", 3))
            self.spots.append({"name": s["name"], "rect": s["rect"], "capacity": capacity,
                               "stock": stock, "low_at": low_at, "armed": False, "gone": 0,
                               "back": 0, "change": 0.0, "taken": 0,
                               "low_alerted": stock <= low_at, "out_alerted": stock == 0})

    # ---- dashboard actions ----
    def calibrate(self):
        self.calibrate_req.set()

    def adjust(self, name, delta):
        with self.lock:
            s = self._find(name)
            if not s:
                return False
            s["stock"] = max(0, min(999, s["stock"] + int(delta)))
            self._persist(s, int(delta), "manual")
            self._stock_alerts(s)
        self._publish()
        return True

    def refill(self, name):
        with self.lock:
            s = self._find(name)
            if not s:
                return False
            delta = s["capacity"] - s["stock"]
            s["stock"] = s["capacity"]
            self._persist(s, delta, "refill")
            self._stock_alerts(s)
        self._publish()
        return True

    def _find(self, name):
        return next((s for s in self.spots if s["name"] == name), None)

    # ---- thread loop ----
    def run(self):
        self._publish()
        while True:
            cap, err = open_capture(self.src, self.is_file)
            if cap is None:
                self.hub.set_status("shelf", "offline", err)
                time.sleep(3)
                continue
            self._set_live_status()
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            if not (1 <= fps <= 120):
                fps = 25.0
            clock, t0, since = 0.0, time.time(), 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    if self.is_file and since > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        since = 0
                        with self.lock:          # video restarts: forget what was on the spots
                            for s in self.spots:
                                s.update(armed=False, gone=0, back=0)
                        continue
                    break
                since += 1
                if self.is_file:
                    clock += 1.0 / fps
                    lag = clock - (time.time() - t0)
                    if lag > 0:
                        time.sleep(min(lag, 0.5))
                auto = self.is_file and self.sc.get("auto_calibrate", True) and since == 1
                self._process(resize_to(frame, self.width), auto_ref=auto)
            cap.release()
            self.hub.set_status("shelf", "offline", "Shelf camera disconnected. Retrying.")
            time.sleep(2)

    def _set_live_status(self):
        msg = ("Watching shelf spots" if self.reference is not None
               else "Clear the shelf spots, then press Calibrate")
        self.hub.set_status("shelf", "live", msg)

    def _process(self, frame, auto_ref=False):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        h, w = gray.shape
        if auto_ref:                         # video: its first frame is the empty shelf
            self.calibrate_req.set()
        if self.calibrate_req.is_set():
            self.calibrate_req.clear()
            self.reference = gray.copy()
            with self.lock:
                for s in self.spots:
                    s.update(armed=False, gone=0, back=0)
            self._set_live_status()
        if self.reference is not None and self.reference.shape != gray.shape:
            self.reference = None            # camera resolution changed: recalibrate
            self._set_live_status()

        diff = cv2.absdiff(self.reference, gray) if self.reference is not None else None
        thr = int(self.sc.get("diff_threshold", 30))
        occ_frac = float(self.sc.get("occupy_frac", 0.06))
        remove_n = int(self.sc.get("remove_frames", 6))
        restock_n = int(self.sc.get("restock_frames", 3))

        with self.lock:
            for s in self.spots:
                x, y, rw, rh = rect_px(s["rect"], w, h)
                if diff is None or rw <= 0 or rh <= 0:
                    s["change"] = 0.0
                    continue
                patch = diff[y:y + rh, x:x + rw]
                frac = float(np.count_nonzero(patch > thr)) / max(1, patch.size)
                s["change"] = frac
                occupied = frac > occ_frac
                if s["armed"]:                          # item on its spot
                    s["gone"] = s["gone"] + 1 if not occupied else 0
                    if s["gone"] >= remove_n:           # picked up
                        s["armed"], s["back"] = False, 0
                        self._take(s)
                else:                                   # waiting for an item
                    s["back"] = s["back"] + 1 if occupied else 0
                    if s["back"] >= restock_n:
                        s["armed"], s["gone"] = True, 0
            self._draw(frame, w, h)
        self._publish()
        self.hub.set_frame("shelf", encode(frame))

    # called with self.lock held
    def _take(self, s):
        s["taken"] += 1
        if s["stock"] > 0:
            s["stock"] -= 1
        self._persist(s, -1, "picked")
        self._stock_alerts(s)

    def _persist(self, s, delta, reason):
        self.hub.log("INSERT INTO inventory_events VALUES (?,?,?,?,?)",
                     (now_iso(), s["name"], delta, s["stock"], reason))
        self.hub.log("INSERT OR REPLACE INTO inventory (item, stock) VALUES (?,?)",
                     (s["name"], s["stock"]))

    def _stock_alerts(self, s):
        if s["stock"] == 0 and not s["out_alerted"]:
            s["out_alerted"] = True
            self.hub.alert("out_of_stock", s["name"], f"{s['name']} is out of stock. Refill the shelf.")
        elif 0 < s["stock"] <= s["low_at"] and not s["low_alerted"]:
            s["low_alerted"] = True
            self.hub.alert("low_stock", s["name"], f"{s['name']} is running low: {s['stock']} left.")
        if s["stock"] > s["low_at"]:
            s["low_alerted"] = False
        if s["stock"] > 0:
            s["out_alerted"] = False

    def _draw(self, frame, w, h):
        for s in self.spots:
            x, y, rw, rh = rect_px(s["rect"], w, h)
            if self.reference is None:
                col = WHITE
            elif s["stock"] == 0:
                col = RED
            else:
                col = MARIGOLD if s["armed"] else TEAL
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), col, 2)
            tag(frame, f"{s['name']}: {s['stock']} left", x + 4, y - 4, col,
                INK if col in (MARIGOLD, WHITE) else WHITE)
        if self.reference is None:
            tag(frame, "Clear the shelf, then press Calibrate on the dashboard",
                12, 30, WHITE, INK, 0.6)

    def _publish(self):
        calibrated = self.reference is not None
        with self.lock:
            spots = [{"name": s["name"], "stock": s["stock"], "capacity": s["capacity"],
                      "low_at": s["low_at"], "taken": s["taken"],
                      "change": round(s["change"] * 100),
                      "state": ("uncalibrated" if not calibrated
                                else "on_shelf" if s["armed"] else "empty")}
                     for s in self.spots]
        self.hub.update("shelf", {"calibrated": calibrated, "spots": spots})
