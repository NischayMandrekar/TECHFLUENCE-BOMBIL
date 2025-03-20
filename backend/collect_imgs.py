import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 4  # Classes 0 to 14
dataset_size = 100  # Capture 20 images per class

cap = cv2.VideoCapture(0)

existing_classes = [int(d) for d in os.listdir(DATA_DIR) if d.isdigit()]

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if j not in existing_classes:
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
