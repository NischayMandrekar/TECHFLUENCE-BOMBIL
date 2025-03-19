import kagglehub
import shutil
import os

# Define the dataset path inside SIGNLANG/datasets/
DATASET_PATH = "SIGNLANG/datasets/wlasl-processed"

# Download the dataset
path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
print("Downloaded dataset files are at:", path)

# Ensure the datasets folder exists
os.makedirs("SIGNLANG/datasets", exist_ok=True)

# Move the dataset to the correct location
if os.path.exists(DATASET_PATH):
    shutil.rmtree(DATASET_PATH)  # Remove old dataset if it exists

shutil.move(path, DATASET_PATH)
print(f"Dataset moved to {DATASET_PATH}")
