import os
import json

# Define dataset path
DATASET_PATH = os.path.join("..", "datasets", "wlasl-processed")  # Adjust path if needed
JSON_FILE = os.path.join(DATASET_PATH, "WLASL_v0.3.json")

# Check if file exists
if not os.path.exists(JSON_FILE):
    raise FileNotFoundError(f"Error: {JSON_FILE} not found. Check dataset folder.")

# Load JSON data
with open(JSON_FILE, "r") as f:
    data = json.load(f)

# Example: Extracting video paths
video_data = []
for entry in data:
    gloss = entry.get("gloss", "unknown")
    instances = entry.get("instances", [])
    
    for instance in instances:
        video_path = instance.get("video_id", "")
        video_data.append({"gloss": gloss, "video": video_path})

# Print sample output
print(f"Loaded {len(video_data)} sign language videos.")
print("Example:", video_data[:5])

# Save preprocessed data
PREPROCESSED_PATH = os.path.join(DATASET_PATH, "preprocessed")
os.makedirs(PREPROCESSED_PATH, exist_ok=True)

OUTPUT_JSON = os.path.join(PREPROCESSED_PATH, "preprocessed_wlasl.json")
with open(OUTPUT_JSON, "w") as out_f:
    json.dump(video_data, out_f, indent=4)

print(f"Preprocessed data saved at {OUTPUT_JSON}")
