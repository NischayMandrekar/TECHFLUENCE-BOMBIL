import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Update the DATA_DIR path
DATA_DIR = './data'

data = []
labels = []

# Iterate over numeric directories (0, 1, 2)
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    if not os.path.exists(dir_path):
        print(f"Warning: Directory {dir_path} does not exist. Skipping...")
        continue
    
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)

        # Ensure the file is an image
        if not img_full_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {img_full_path}")
            continue

        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Error: Could not read image {img_full_path}. Skipping...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []

            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Save the processed data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset creation completed successfully!")