import pandas as pd

df = pd.read_csv("queue_tracking_data.csv")

people_per_frame = df.groupby("frame")["track_id"].nunique()

threshold = 18

total_frames = len(people_per_frame)
average_people = people_per_frame.mean()
max_people = people_per_frame.max()

crowded_frames = people_per_frame[people_per_frame >= threshold]

print("========== QUEUE/CROWD REPORT ==========")

print("Total frames:", total_frames)
print("Average people:", round(average_people, 2))
print("Maximum people:", max_people)
print("Crowded frames:", len(crowded_frames))

if len(crowded_frames) > 0:
    percentage = (len(crowded_frames) / total_frames) * 100

    print("Crowded percentage:", round(percentage, 2), "%")
    print("First crowded frame:", crowded_frames.index[0])
else:
    print("No overcrowding detected.")

print("========================================")