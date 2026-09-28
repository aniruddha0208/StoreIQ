import pandas as pd

df = pd.read_csv("queue_tracking_data.csv")

people_per_frame = df.groupby("frame")["track_id"].nunique()

threshold = 18
required_frames = 15

consecutive = 0
alerts = []

for frame, count in people_per_frame.items():

    if count >= threshold:
        consecutive += 1

        if consecutive == required_frames:
            alerts.append({
                "frame": frame,
                "people": count,
                "alert": "OVERCROWDING"
            })

    else:
        consecutive = 0

alerts_df = pd.DataFrame(alerts)

alerts_df.to_csv("alerts.csv", index=False)

print("Alerts generated:", len(alerts_df))

if len(alerts_df) > 0:
    print("\nAlert log:")
    print(alerts_df)
else:
    print("No sustained overcrowding alerts.")