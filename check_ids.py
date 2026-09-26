from pathlib import Path
import re

track_folder = Path(r"C:\shopper_ai\runs\detect\track")

ids = set()

for image_file in track_folder.glob("*.jpg"):
    # This version only checks filenames for now.
    # We will get actual tracking IDs from YOLO's tracking output next.
    pass

print("Tracking output folder found:", track_folder.exists())
print("Processed frames:", len(list(track_folder.glob("*.jpg"))))