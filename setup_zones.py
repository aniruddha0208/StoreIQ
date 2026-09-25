"""Draw zones and save them to config.json.

    python setup_zones.py          # floor sections + checkout queue, then shelf spots
    python setup_zones.py floor    # only the floor video
    python setup_zones.py shelf    # only the shelf camera

Drawing: drag a box, press ENTER (or SPACE) to keep it, draw the next one,
press ESC when you are done. Names are typed in the terminal afterwards.
"""

import json
import os
import sys

import cv2

from vision import TEAL, resize_to, resolve_source

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(BASE_DIR, "config.json")


def grab_frame(source, width):
    src, is_file = resolve_source(source, BASE_DIR)
    if is_file and not os.path.exists(src):
        sys.exit(f"Video not found: {src}\nPut your video next to app.py or fix 'source' in config.json.")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open {source!r}. For a webcam, try 0, 1 or 2 in config.json.")
    frame = None
    for _ in range(1 if is_file else 30):   # webcams need a moment to set exposure
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        sys.exit("Could not read a frame from the source.")
    return resize_to(frame, width)


def norm(r, w, h):
    x, y, bw, bh = (int(v) for v in r)
    return [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)]


def ask(prompt, default):
    v = input(f"  {prompt} [{default}]: ").strip()
    return v or str(default)


def ask_int(prompt, default):
    while True:
        v = ask(prompt, default)
        try:
            return int(v)
        except ValueError:
            print("  Please type a whole number.")


def draw_many(frame, title):
    rois = cv2.selectROIs(title, frame, showCrosshair=False)
    cv2.destroyWindow(title)
    cv2.waitKey(1)
    return [r for r in (rois if len(rois) else []) if r[2] > 0 and r[3] > 0]


def setup_floor(cfg):
    fc = cfg["floor"]
    frame = grab_frame(fc.get("source", 0), cfg.get("process_width", 960))
    h, w = frame.shape[:2]
    old = fc.get("sections", [])

    print("\nFLOOR: draw one box per shop section (ENTER after each, ESC when done).")
    rects = draw_many(frame, "Draw shop sections: ENTER after each box, ESC when done")
    if rects:
        sections = []
        for i, r in enumerate(rects):
            default = old[i]["name"] if i < len(old) else f"Section {chr(65 + i)}"
            sections.append({"name": ask(f"Name for section {i + 1}", default), "rect": norm(r, w, h)})
        fc["sections"] = sections
    else:
        print("  No boxes drawn; keeping the existing sections.")

    q = fc.get("queue") or {"name": "Checkout", "rect": None, "alert_at": 4, "alert_after_s": 5}
    if ask("Draw a checkout queue zone? (y/n)", "y").lower().startswith("y"):
        preview = frame.copy()
        for s in fc["sections"]:
            x, y, bw, bh = (int(s["rect"][0] * w), int(s["rect"][1] * h),
                            int(s["rect"][2] * w), int(s["rect"][3] * h))
            cv2.rectangle(preview, (x, y), (x + bw, y + bh), TEAL, 2)
        print("  Drag a box over the checkout queue, press ENTER. Press C to skip.")
        r = cv2.selectROI("Draw the checkout queue zone, then ENTER", preview, showCrosshair=False)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if r[2] > 0 and r[3] > 0:
            q["rect"] = norm(r, w, h)
            q["name"] = ask("Name for the queue", q.get("name", "Checkout"))
            q["alert_at"] = ask_int("Alert when this many people are waiting", q.get("alert_at", 4))
    else:
        q["rect"] = None
    fc["queue"] = q


def setup_shelf(cfg):
    sc = cfg["shelf"]
    frame = grab_frame(sc.get("source", 0), cfg.get("process_width", 960))
    h, w = frame.shape[:2]
    old = sc.get("spots", [])

    print("\nSHELF: draw one box per product spot (ENTER after each, ESC when done).")
    rects = draw_many(frame, "Draw shelf spots: ENTER after each box, ESC when done")
    if not rects:
        print("  No boxes drawn; keeping the existing shelf spots.")
        return
    spots = []
    for i, r in enumerate(rects):
        prev = old[i] if i < len(old) else {}
        print(f"  Spot {i + 1}:")
        spots.append({"name": ask("Product name", prev.get("name", "Chips" if i == 0 else f"Item {i + 1}")),
                      "rect": norm(r, w, h),
                      "stock": ask_int("Full stock count", prev.get("stock", 10)),
                      "low_at": ask_int("Low-stock alert at", prev.get("low_at", 3))})
    sc["spots"] = spots


def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if target not in ("all", "floor", "shelf"):
        sys.exit("Usage: python setup_zones.py [floor|shelf]")
    with open(CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    if target in ("all", "floor"):
        setup_floor(cfg)
    if target in ("all", "shelf"):
        setup_shelf(cfg)
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("\nSaved to config.json. Now run:  python app.py")


if __name__ == "__main__":
    main()
