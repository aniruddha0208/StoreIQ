"""Local SQLite log (offline-first). Survives damage: a corrupt file is moved
aside automatically and a fresh one is started, so logging can never crash the demo."""

import datetime
import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS inventory        (item TEXT PRIMARY KEY, stock INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS inventory_events (ts TEXT, item TEXT, delta INTEGER, stock_after INTEGER, reason TEXT);
CREATE TABLE IF NOT EXISTS visits           (ts TEXT, section TEXT, track_id INTEGER);
CREATE TABLE IF NOT EXISTS dwell            (ts TEXT, section TEXT, track_id INTEGER, seconds REAL);
CREATE TABLE IF NOT EXISTS queue_readings   (ts TEXT, zone TEXT, length INTEGER);
CREATE TABLE IF NOT EXISTS alerts           (ts TEXT, kind TEXT, zone TEXT, message TEXT);
"""


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _is_corrupt(err):
    msg = str(err).lower()
    return "malformed" in msg or "not a database" in msg or "disk image" in msg


class Store:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.con = None
        self.last_error = ""
        self._open()

    # ---------- connection management ----------
    def _connect(self):
        con = sqlite3.connect(self.path, check_same_thread=False, timeout=5)
        con.execute("PRAGMA journal_mode=WAL")      # safer against interrupted writes
        con.execute("PRAGMA synchronous=NORMAL")
        con.executescript(SCHEMA)
        row = con.execute("PRAGMA quick_check").fetchone()
        if not row or row[0] != "ok":
            con.close()
            raise sqlite3.DatabaseError("database disk image is malformed")
        return con

    def _open(self):
        try:
            self.con = self._connect()
            return
        except sqlite3.DatabaseError as e:
            self.last_error = str(e)
            self._quarantine()
        try:
            self.con = self._connect()
            self.last_error = ""
        except sqlite3.DatabaseError as e:
            self.con = None
            self.last_error = str(e)
            print(f"[log] logging turned off: {e}")

    def _quarantine(self):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for suffix in ("", "-wal", "-shm"):
            p = self.path + suffix
            if os.path.exists(p):
                try:
                    os.replace(p, f"{self.path}.corrupt-{stamp}{suffix}")
                except OSError:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        print(f"[log] {os.path.basename(self.path)} was damaged; moved it aside and started a fresh one")

    # ---------- public API ----------
    def execute(self, sql, params=()):
        with self.lock:
            if self.con is None:
                return False
            try:
                self.con.execute(sql, params)
                self.con.commit()
                return True
            except sqlite3.DatabaseError as e:
                self.last_error = str(e)
                print(f"[log] write skipped: {e}")
                if _is_corrupt(e):
                    try:
                        self.con.close()
                    except Exception:
                        pass
                    self.con = None
                    self._quarantine()
                    try:
                        self.con = self._connect()
                        self.last_error = ""
                    except sqlite3.DatabaseError as e2:
                        self.last_error = str(e2)
                return False

    def query(self, sql, params=()):
        with self.lock:
            if self.con is None:
                return []
            try:
                return self.con.execute(sql, params).fetchall()
            except sqlite3.DatabaseError as e:
                self.last_error = str(e)
                return []

    def health(self):
        name = os.path.basename(self.path)
        if self.con is None:
            return {"state": "error", "msg": f"Not saving: {self.last_error}"}
        return {"state": "live", "msg": f"Saving to {name}"}
