import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start the video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hand detection
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

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
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Extract hand landmarks and normalize them
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize landmarks to fit into the input format expected by the model
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure the model receives the correct number of features (84)
        if len(data_aux) != 84:
            print(f"Expected 84 features, but got {len(data_aux)}. Adjusting.")
            continue  # Skip this iteration if the data is invalid

        # Predict the sign language character
        prediction = model.predict([np.asarray(data_aux)])

        # Get the predicted character from the model
        predicted_character = labels_dict[int(prediction[0])]

        # Draw the bounding box and display the predicted character
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Show the processed frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
