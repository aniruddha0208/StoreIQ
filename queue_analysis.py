import pandas as pd

df = pd.read_csv("queue_tracking_data.csv")

people_per_frame = df.groupby("frame")["track_id"].nunique()

print("Frames:", len(people_per_frame))
print("Average:", round(people_per_frame.mean(), 2))
print("Maximum:", people_per_frame.max())
print("Minimum:", people_per_frame.min())

print("\nPeople count distribution:")
print(people_per_frame.value_counts().sort_index())