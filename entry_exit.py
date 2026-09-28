import pandas as pd

# Load tracking data
df = pd.read_csv("video_tracking_data.csv")

entries = 0
exits = 0

print("Entry/Exit analysis:\n")

for track_id, person in df.groupby("track_id"):

    # Sort this person's detections by frame
    person = person.sort_values("frame")

    # Center Y position
    first_y = (person.iloc[0]["y1"] + person.iloc[0]["y2"]) / 2
    last_y = (person.iloc[-1]["y1"] + person.iloc[-1]["y2"]) / 2

    movement = last_y - first_y

    # Ignore very small movements
    if abs(movement) < 50:
        continue

    if movement < 0:
        entries += 1
        direction = "TOWARDS CAMERA"

    else:
        exits += 1
        direction = "AWAY FROM CAMERA"

    print(f"ID {track_id}: {direction} (movement = {movement:.1f})")

print("\n-------------------------")
print("ENTRY COUNT:", entries)
print("EXIT COUNT:", exits)
print("-------------------------")