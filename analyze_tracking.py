import pandas as pd

df = pd.read_csv("tracking_data.csv")

unique_people = df["track_id"].nunique()

print("Total tracking records:", len(df))
print("Unique tracking IDs:", unique_people)
print("Frames processed:", df["frame"].nunique())

print("\nFirst 10 tracking records:")
print(df.head(10))