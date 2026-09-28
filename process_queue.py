import cv2
import csv
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

source = r"C:\shopper_ai\data\queue.mp4"

output_csv = "queue_results.csv"

threshold = 18
required_frames = 15
consecutive = 0
alert_active = False

results = model.track(
    source=source,
    tracker="bytetrack.yaml",
    classes=[0],
    stream=True,
    save=False
)

with open(output_csv, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["frame", "people", "status"])

    for frame_number, result in enumerate(results, start=1):

        frame = result.orig_img.copy()

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)

            people = len(ids)

            for box, track_id in zip(boxes, ids):

                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        else:
            people = 0

        if people >= threshold:
            consecutive += 1
        else:
            consecutive = 0
            alert_active = False

        if consecutive >= required_frames:
            alert_active = True

        if alert_active:
            status = "OVERCROWDING"

        elif people >= 14:
            status = "CROWDED"

        else:
            status = "NORMAL"

        writer.writerow([
            frame_number,
            people,
            status
        ])

        cv2.putText(
            frame,
            f"People: {people}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if alert_active else (0, 255, 0),
            3
        )

        cv2.imshow("Retail AI Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()

print("Processing complete.")
print("Results saved to:", output_csv)