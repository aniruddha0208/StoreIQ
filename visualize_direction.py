import cv2
import pandas as pd

video_path = r"C:\shopper_ai\data\mall.mp4"
csv_path = "video_tracking_data.csv"
output_path = "direction_output.mp4"

df = pd.read_csv(csv_path)

# Calculate center Y
df["center_y"] = (df["y1"] + df["y2"]) / 2

# Calculate direction for each ID
directions = {}

for track_id, person in df.groupby("track_id"):

    person = person.sort_values("frame")

    if len(person) < 10:
        directions[track_id] = "UNKNOWN"
        continue

    n = len(person)
    window = max(3, int(n * 0.3))

    start_y = person["center_y"].iloc[:window].mean()
    end_y = person["center_y"].iloc[-window:].mean()

    movement = end_y - start_y

    if movement > 50:
        directions[track_id] = "AWAY"

    elif movement < -50:
        directions[track_id] = "TOWARDS"

    else:
        directions[track_id] = "STATIONARY"


cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    current = df[df["frame"] == frame_number]

    for _, row in current.iterrows():

        track_id = int(row["track_id"])

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        direction = directions.get(track_id, "UNKNOWN")

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        label = f"ID {track_id}: {direction}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    out.write(frame)

cap.release()
out.release()

print("Visualization complete.")
print("Saved to:", output_path)