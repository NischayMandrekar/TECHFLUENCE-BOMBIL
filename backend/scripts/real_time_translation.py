import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import os
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model(os.path.join('models', 'asl_translation_model.h5'))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Load label encoder classes
label_classes = np.load(os.path.join('data', 'y_labels.npy'), allow_pickle=True)

# Recreate the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# Function to extract landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks.extend([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
    return np.array(landmarks).flatten()

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract landmarks and predict
    landmarks = extract_landmarks(frame)
    if landmarks.size > 0:
        landmarks = landmarks.reshape(1, 1, landmarks.shape[0])  # Reshape for LSTM input
        prediction = model.predict(landmarks)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        
        # Display the predicted label as captions
        cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert the predicted label to speech
        engine.say(predicted_label)
        engine.runAndWait()

    # Display the frame
    cv2.imshow('Sign Language Translation', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()