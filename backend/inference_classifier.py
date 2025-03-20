import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand detection with optimized confidence values
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define labels for predictions
labels_dict = {
    2: 'drink', 5: 'i love you', 7: 'me', 8: 'no', 
    11: 'thank you', 12: 'warning', 13: 'yes', 14: 'you', 
    15: 'hello', 16: 'goodbye', 17: 'please', 18: 'sorry', 
    24: 'understand', 25: 'think', 26: 'this', 27: 'who', 
    28: 'why', 30: 'look'
}

while True:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    H, W, _ = frame.shape  # Get frame dimensions

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        print("[INFO] Hand detected!")  # Debug message

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract hand landmarks and normalize them
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Ensure x_ and y_ are not empty before processing
            if not x_ or not y_:
                print("[WARNING] Empty x_ or y_ lists. Skipping frame.")
                continue

            min_x, min_y = min(x_), min(y_)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

        # Ensure correct input size for the model (84 features)
        if len(data_aux) == 84:
            # Convert to NumPy array for prediction
            prediction = model.predict([np.asarray(data_aux)])[0]
            predicted_character = labels_dict.get(int(prediction), "Unknown")

            print(f"[INFO] Prediction: {predicted_character}")  # Debugging output

            # Draw bounding box and display the predicted character
            x1, y1 = int(min(x_) * W) - 20, int(min(y_) * H) - 20
            x2, y2 = int(max(x_) * W) + 20, int(max(y_) * H) + 20
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            print(f"[WARNING] Expected 84 features, but got {len(data_aux)}. Skipping frame.")

    else:
        print("[INFO] No hand detected.")  # Debugging output

    # Show the processed frame
    cv2.imshow('Hand Sign Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
