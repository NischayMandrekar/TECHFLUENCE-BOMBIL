import cv2
import torch
import numpy as np
from train import SignLanguageModel

# Load the trained model
MODEL_SAVE_PATH = "SIGNLANG/models/sign_model.pth"
model = SignLanguageModel()
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# Function to process webcam input
def recognize_sign():
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame (resize to model's input size)
        frame_resized = cv2.resize(frame, (224, 224))
        frame_tensor = torch.tensor(frame_resized).unsqueeze(0).float()  # Add batch dimension
        
        # Perform inference
        with torch.no_grad():
            output = model(frame_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        # Display result on the screen
        cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Real-time Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_sign()
