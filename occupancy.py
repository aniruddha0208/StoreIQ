import pandas as pd

df = pd.read_csv("video_tracking_data.csv")

occupancy = df.groupby("frame")["track_id"].nunique()

print("Average people per frame:", round(occupancy.mean(), 2))
print("Maximum people in a frame:", occupancy.max())
print("Minimum people in a frame:", occupancy.min())

print("\nFirst 20 frames:")
print(occupancy.head(20))