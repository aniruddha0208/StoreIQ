from ultralytics import YOLO
import csv

model = YOLO("yolo11n.pt")

source = r"C:\Users\mahaj\Downloads\MOT17\MOT17\train\MOT17-09-SDP\img1"

output_file = "tracking_data.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2"])

    results = model.track(
        source=source,
        tracker="bytetrack.yaml",
        classes=[0],
        stream=True
    )

    for frame_number, result in enumerate(results, start=1):

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box

            writer.writerow([
                frame_number,
                track_id,
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ])

print("Tracking complete.")
print("Data saved to:", output_file)