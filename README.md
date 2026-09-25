# SIH26179 Retail Intelligence — working prototype

One app, two cameras, one live dashboard. Everything runs on your laptop; no internet
needed after the first install.

| Feature | Camera | How it works |
|---|---|---|
| Shelf stock | Webcam on the table | Watches each product spot. Picking an item up lowers its stock by 1. No ML, so it works at any angle. |
| Footfall by section | Store video (or a camera) | YOLO finds people, a tracker follows them, and each section counts visitors and average dwell time. |
| Checkout queue | Same store video | Counts people in the queue zone and alerts when it stays too long. |
| Alerts | — | Low stock, out of stock, long queue. Shown live and saved. |
| Dashboard | — | http://localhost:8000: live feeds, ranking, trend, stock controls, CSV report. |

## 0. Test videos included

`store.mp4` and `shelf.mp4` in this folder are **synthetic test clips**, so you can run everything right away:

- `shelf.mp4`: a red box appears on the table spot and is removed 3 times per loop. This uses the real
  shelf logic, exactly as it would work on your own video.
- `store.mp4`: **white rectangles stand in for people.** Real YOLO cannot see rectangles as people, so
  config.json ships with `"detector": "test_blobs"`. The dashboard labels this as test mode.

**For the real demo:** replace `store.mp4` with real store footage, set `floor.detector` back to `"yolo"`,
and run `python setup_zones.py floor` to redraw the sections and queue on it. Replace `shelf.mp4` with
your own recording of the chips (see step 2) and run `python setup_zones.py shelf`.

## 1. Install (once)

With uv:
```
uv venv
uv pip install -r requirements.txt
```
Or with pip: `pip install -r requirements.txt`

The first run downloads the YOLO weights (about 20 MB), so be online the first time.

## 2. Add your two videos (no cameras needed)

Put both clips in this folder:

- `store.mp4`: shop floor footage with people walking (footfall + queue)
- `shelf.mp4`: a phone video of the table where you pick up the chips packet

How to record `shelf.mp4` (about 30 s, phone held still or propped up):
1. Start with the table **empty** for 2 to 3 seconds. The first frame is used as the empty shelf, so no calibrate click is needed.
2. Put the packet down, wait 2 s, pick it up, wait 2 s. Repeat 3 or 4 times.
3. End with the table empty.

Each video loops, and every pickup lowers the stock by one. Use Refill on the dashboard to reset it.
To use a live camera instead, set `source` to `0` in config.json.

## 3. Draw your zones (recommended)

```
uv run python setup_zones.py
```
1. On the store video: drag a box per shop section, press ENTER after each, then ESC. Type names in the terminal.
2. Draw the checkout queue zone (or press C to skip).
3. On the shelf video: drag a box over each product spot, press ENTER after each, then ESC. Enter name, full stock and low-stock level.

The app also runs without this step, using three equal default sections and one shelf spot in the middle.

## 4. Run

```
uv run python app.py
```
The dashboard opens at http://localhost:8000.

With `shelf.mp4` the shelf calibrates itself from the first frame. (Only a live camera needs the
**Calibrate empty shelf** button.)

## Demo script (about 3 minutes)

1. Store video playing: point at the section boxes, the live "now" counts and the ranking.
2. Busiest section is highlighted in yellow on both the video and the ranking.
3. Queue builds up: the queue number turns red and an alert appears.
4. The chips packet is picked up in the shelf video: stock drops, and a low-stock alert fires at the set level.
5. Click Refill, then Download report (CSV).
6. Privacy point: no video or images are saved. Only counts and times go into store.db.

## Tuning (config.json)

| Problem | Change |
|---|---|
| People in the video are missed | lower `floor.conf` (0.2), or `yolo_model` to `yolov8m.pt` |
| Floor video is slow or laggy | set `process_every` to 2 or 3, or `yolo_model` to `yolov8n.pt` |
| One person counted as several visitors | raise `tracking.max_dist` (0.12) and `max_missed` (20) |
| Shelf fires with nothing picked | raise `shelf.occupy_frac` or `diff_threshold` |
| Pickup not noticed | lower `shelf.occupy_frac`; recalibrate with spots empty |
| Shelf video not found | name it `shelf.mp4` or fix `shelf.source` |
| Only want footfall | set `shelf.enabled` to false |
| Port 8000 busy | change `port` |
| Show dashboard on another device | set `host` to `"0.0.0.0"` and open `http://<laptop-ip>:8000` |

## Notes

- `store.db` is created automatically. If it ever gets damaged, it is moved aside and a
  fresh one starts, so logging can't crash the demo. Keep this folder **out of OneDrive,
  Google Drive or Dropbox**; sync apps are the usual cause of damaged SQLite files.
- If `ultralytics` is missing, the floor falls back to OpenCV's HOG person detector (weaker, but it runs).
- Record `shelf.mp4` with the phone held still. Camera shake looks like movement on the spot.

## Files

```
app.py            web server, dashboard API, starts both pipelines
vision.py         floor (people, sections, queue) and shelf (pickup) pipelines
store.py          local SQLite log with automatic recovery
setup_zones.py    draw sections, queue and shelf spots into config.json
config.json       all settings and zones
static/index.html the dashboard (no internet needed)
```
