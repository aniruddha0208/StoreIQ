import pandas as pd

df = pd.read_csv("queue_tracking_data.csv")

people_per_frame = df.groupby("frame")["track_id"].nunique()

print("Average people per frame:", round(people_per_frame.mean(), 2))
print("Maximum people in a frame:", people_per_frame.max())
print("Minimum people in a frame:", people_per_frame.min())

print("\nFirst 20 frames:")
print(people_per_frame.head(20))
