import pandas as pd

df = pd.read_csv("video_tracking_data.csv")

# Calculate center Y position of each person
df["center_y"] = (df["y1"] + df["y2"]) / 2

# Sort properly
df = df.sort_values(["track_id", "frame"])

print("Movement direction of tracked people:\n")

towards_camera = 0
away_from_camera = 0
stationary = 0

for track_id, person in df.groupby("track_id"):

    # Ignore very short tracks
    if len(person) < 10:
        continue

    # Use first and last 30% of the track
    n = len(person)
    window = max(3, int(n * 0.3))

    start_y = person["center_y"].iloc[:window].mean()
    end_y = person["center_y"].iloc[-window:].mean()

    movement = end_y - start_y

    if movement > 50:
        direction = "AWAY"
        away_from_camera += 1

    elif movement < -50:
        direction = "TOWARDS"
        towards_camera += 1

    else:
        direction = "STATIONARY"
        stationary += 1

    print(
        f"ID {track_id}: "
        f"{direction} "
        f"(movement = {movement:.1f})"
    )

print("\n-------------------------")
print("TOWARDS CAMERA:", towards_camera)
print("AWAY FROM CAMERA:", away_from_camera)
print("STATIONARY:", stationary)
print("-------------------------")