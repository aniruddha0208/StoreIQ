"""SIH26179 Retail Intelligence prototype.

Runs the shelf-pickup and floor-footfall pipelines in the background and serves a
live dashboard. Everything runs locally; no internet needed after the first install.

    python app.py            (or: uv run python app.py)
    then open http://localhost:8000
"""

import csv
import io
import json
import logging
import os
import threading
import time
import webbrowser
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from store import Store, now_iso
from vision import FloorPipeline, ShelfPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(BASE_DIR, "config.json"), encoding="utf-8") as f:
        return json.load(f)


class Hub:
    """Shared, thread-safe state between the pipelines and the web server."""

    def __init__(self, store, bucket_s):
        self.store = store
        self.bucket = max(1, int(bucket_s))
        self.lock = threading.Lock()
        self.frames = {}
        self.status = {}
        self.data = {"floor": None, "shelf": None}
        self.alerts = deque(maxlen=40)
        self.trend = {}
        self.started = time.time()

    def set_status(self, name, state, msg=""):
        with self.lock:
            self.status[name] = {"state": state, "msg": msg}

    def set_frame(self, name, jpeg):
        if jpeg:
            self.frames[name] = jpeg

    def update(self, name, data):
        with self.lock:
            self.data[name] = data

    def log(self, sql, params=()):
        if self.store:
            self.store.execute(sql, params)

    def alert(self, kind, zone, message):
        item = {"ts": now_iso(), "kind": kind, "zone": zone, "message": message}
        with self.lock:
            self.alerts.appendleft(item)
        self.log("INSERT INTO alerts VALUES (?,?,?,?)", (item["ts"], kind, zone, message))
        print(f"[alert] {message}")

    def bump_trend(self):
        b = int(time.time() // self.bucket)
        with self.lock:
            self.trend[b] = self.trend.get(b, 0) + 1
            for old in [k for k in self.trend if k < b - 120]:
                del self.trend[old]

    def clear_trend(self):
        with self.lock:
            self.trend.clear()

    def saved_stock(self):
        rows = self.store.query("SELECT item, stock FROM inventory") if self.store else []
        return {r[0]: int(r[1]) for r in rows}

    def snapshot(self, points=30):
        cur = int(time.time() // self.bucket)
        with self.lock:
            trend = [{"t": (cur - points + 1 + i) * self.bucket,
                      "v": self.trend.get(cur - points + 1 + i, 0)} for i in range(points)]
            return {"status": dict(self.status), "floor": self.data["floor"],
                    "shelf": self.data["shelf"], "alerts": list(self.alerts), "trend": trend,
                    "log": self.store.health() if self.store else {"state": "off", "msg": "Logging turned off"},
                    "uptime_s": int(time.time() - self.started)}


def placeholder(text):
    img = np.full((360, 640, 3), (58, 52, 39), np.uint8)
    cv2.putText(img, text, (28, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (215, 220, 215), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def create_app(cfg, hub, floor, shelf):
    app = Flask(__name__, static_folder=None)
    static_dir = os.path.join(BASE_DIR, "static")
    holders = {"floor": placeholder("Waiting for the floor video"),
               "shelf": placeholder("Waiting for the shelf camera")}

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/state")
    def state():
        s = hub.snapshot()
        s["store_name"] = cfg.get("store_name", "Store")
        s["trend_bucket_s"] = hub.bucket
        return jsonify(s)

    @app.get("/stream/<name>")
    def stream(name):
        if name not in holders:
            return "Unknown stream", 404

        def gen():
            last = None
            while True:
                f = hub.frames.get(name) or holders[name]
                if f is not last:
                    last = f
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(f)).encode() + b"\r\n\r\n" + f + b"\r\n")
                time.sleep(0.04)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    def body():
        return request.get_json(silent=True) or {}

    def shelf_off():
        return jsonify({"ok": False, "error": "The shelf camera is turned off in config.json"}), 409

    @app.post("/api/shelf/calibrate")
    def calibrate():
        if not shelf:
            return shelf_off()
        shelf.calibrate()
        return jsonify({"ok": True})

    @app.post("/api/shelf/adjust")
    def adjust():
        if not shelf:
            return shelf_off()
        b = body()
        return jsonify({"ok": shelf.adjust(b.get("name"), int(b.get("delta", 0)))})

    @app.post("/api/shelf/refill")
    def refill():
        if not shelf:
            return shelf_off()
        return jsonify({"ok": shelf.refill(body().get("name"))})

    @app.post("/api/floor/reset")
    def reset_floor():
        if not floor:
            return jsonify({"ok": False, "error": "The floor camera is turned off in config.json"}), 409
        floor.reset_req.set()
        return jsonify({"ok": True})

    @app.get("/api/report.csv")
    def report():
        s = hub.snapshot()
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["Report generated", now_iso()])
        w.writerow([])
        w.writerow(["Section", "Visitors", "Average dwell (s)", "In section now"])
        for sec in (s["floor"] or {}).get("sections", []):
            w.writerow([sec["name"], sec["visits"], sec["avg_dwell"] if sec["avg_dwell"] is not None else "", sec["now"]])
        w.writerow([])
        w.writerow(["Item", "Stock", "Capacity", "Picked this session"])
        for sp in (s["shelf"] or {}).get("spots", []):
            w.writerow([sp["name"], sp["stock"], sp["capacity"], sp["taken"]])
        w.writerow([])
        w.writerow(["Alert time", "Type", "Where", "Message"])
        for a in s["alerts"]:
            w.writerow([a["ts"], a["kind"], a["zone"], a["message"]])
        return Response(out.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=store_report.csv"})

    return app


def main():
    cfg = load_config()
    store = Store(os.path.join(BASE_DIR, cfg["db"])) if cfg.get("db") else None
    hub = Hub(store, cfg.get("trend_bucket_s", 10))

    floor = shelf = None
    if cfg.get("floor", {}).get("enabled", True):
        floor = FloorPipeline(cfg, hub, BASE_DIR)
        hub.set_status("floor", "starting", "Loading the person detector")
        floor.start()
    else:
        hub.set_status("floor", "off", "Turned off in config.json")
    if cfg.get("shelf", {}).get("enabled", True):
        shelf = ShelfPipeline(cfg, hub, BASE_DIR)
        hub.set_status("shelf", "starting", "Opening the shelf camera")
        shelf.start()
    else:
        hub.set_status("shelf", "off", "Turned off in config.json")

    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # keep the terminal readable
    host = cfg.get("host", "127.0.0.1")
    port = int(cfg.get("port", 8000))
    url = f"http://localhost:{port}"
    print(f"\n  Dashboard running at {url}\n  Press Ctrl+C to stop.\n")
    if cfg.get("open_browser", True):
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    create_app(cfg, hub, floor, shelf).run(host=host, port=port, threaded=True,
                                           debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
