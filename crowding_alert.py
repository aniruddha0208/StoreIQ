import pandas as pd

df = pd.read_csv("queue_tracking_data.csv")

people_per_frame = df.groupby("frame")["track_id"].nunique()

threshold = 18
required_frames = 15

consecutive = 0
alert_frame = None

for frame, count in people_per_frame.items():

    if count >= threshold:
        consecutive += 1

        if consecutive == required_frames:
            alert_frame = frame
            break

    else:
        consecutive = 0

if alert_frame:
    print("OVERCROWDING ALERT!")
    print("Alert triggered at frame:", alert_frame)
    print("People at alert:", people_per_frame[alert_frame])
    print("Condition:", f"{threshold}+ people for {required_frames} consecutive frames")

else:
    print("No sustained overcrowding detected.")