import cv2
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

source = r"C:\shopper_ai\data\queue.mp4"

threshold = 18
required_frames = 15

consecutive_crowded = 0
alert_active = False

results = model.track(
    source=source,
    tracker="bytetrack.yaml",
    classes=[0],
    stream=True
)

for result in results:

    frame = result.orig_img.copy()
    count = 0

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        count = len(ids)

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

    # Check sustained overcrowding
    if count >= threshold:
        consecutive_crowded += 1
    else:
        consecutive_crowded = 0
        alert_active = False

    if consecutive_crowded >= required_frames:
        alert_active = True

    # Status
    if alert_active:
        status = "OVERCROWDING ALERT"
        status_color = (0, 0, 255)

    elif count >= 14:
        status = "CROWDED"
        status_color = (0, 165, 255)

    else:
        status = "NORMAL"
        status_color = (0, 255, 0)

    # People count
    cv2.putText(
        frame,
        f"People: {count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Status
    cv2.putText(
        frame,
        status,
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        3
    )

    # Show sustained frame count
    cv2.putText(
        frame,
        f"Threshold: {threshold}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Queue Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()