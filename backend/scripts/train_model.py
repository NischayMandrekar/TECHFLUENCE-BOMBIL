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
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,  # Lowered confidence threshold for better detection
    min_tracking_confidence=0.7    # Increased tracking confidence
)

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image
    image = cv2.resize(image, (320, 320))
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Convert back to BGR for MediaPipe compatibility
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return image

# Function to extract hand landmarks from an image
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks.extend([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
        return np.array(landmarks).flatten()
    return None  # No landmarks found

# Function to visualize landmarks on an image
def visualize_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return image

# Function to load the dataset
def load_dataset(data_dir):
    X = []
    y = []
    skipped_images = []  # To log skipped images

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):  # Ensure it's a directory
            print(f"Loading images from: {label_dir}")
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                image = cv2.imread(file_path)
                if image is not None:
                    image = preprocess_image(image)  # Preprocess the image
                    landmarks = extract_landmarks(image)
                    if landmarks is not None:  # Only add if landmarks are detected
                        X.append(landmarks)
                        y.append(label)
                    else:
                        print(f"No landmarks detected in: {file_path}")
                        skipped_images.append(file_path)  # Log skipped images
                        # Uncomment below to visualize the image for debugging
                        # debug_image = visualize_landmarks(image)
                        # cv2.imshow('Debug Image', debug_image)
                        # cv2.waitKey(0)  # Press any key to continue
                else:
                    print(f"Failed to load image: {file_path}")
    
    # Save the list of skipped images for inspection
    with open(os.path.join('data', 'skipped_images.txt'), 'w') as f:
        for img_path in skipped_images:
            f.write(f"{img_path}\n")
    
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