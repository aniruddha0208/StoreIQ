from ultralytics import YOLO
import csv

model = YOLO("yolo11m.pt")

source = r"C:\shopper_ai\data\mall.mp4"

output_file = "video_tracking_data.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2"])

    results = model.track(
        source=source,
        tracker="bytetrack.yaml",
        classes=[0],
        stream=True,
        save=True
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

print("Video tracking complete.")
print("Data saved to:", output_file)