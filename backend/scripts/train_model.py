import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib  # For saving the label encoder

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Function to extract hand landmarks from an image
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks.extend([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
    return np.array(landmarks).flatten()

# Function to load the dataset
def load_dataset(data_dir):
    X = []
    y = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):  # Ensure it's a directory
            print(f"Loading images from: {label_dir}")
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                image = cv2.imread(file_path)
                if image is not None:
                    landmarks = extract_landmarks(image)
                    if landmarks.size > 0:  # Only add if landmarks are detected
                        X.append(landmarks)
                        y.append(label)
                    else:
                        print(f"No landmarks detected in: {file_path}")
                else:
                    print(f"Failed to load image: {file_path}")
    return np.array(X), np.array(y)

# Load the dataset
data_dir = os.path.join('data', 'asl_dataset')  # Path to the dataset
X, y = load_dataset(data_dir)

# Save the preprocessed data
np.save(os.path.join('data', 'X_landmarks.npy'), X)
np.save(os.path.join('data', 'y_labels.npy'), y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label_encoder
joblib.dump(label_encoder, os.path.join('data', 'label_encoder.pkl'))

# Reshape X for LSTM input (samples, timesteps, features)
X = X.reshape(X.shape[0], 1, X.shape[1])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save the model
model.save(os.path.join('models', 'asl_translation_model.h5'))