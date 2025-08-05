import cv2
import time
import threading
import pygame
import torch
import torch.nn as neural
import timm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Initialize pygame for sound
pygame.mixer.init()
pygame.mixer.music.load("Merry Go Round of Life.mp3")

# Threading setup
lock = threading.Lock()
currently_playing = False

last_eyes_open_time = time.time()
last_eyes_closed_time = time.time()

eyes_closed_confirmed = False
eyes_open_confirmed = True

EYES_CLOSED_THRESHOLD = 0.7  # seconds needed to confirm eyes are really closed
EYES_OPEN_THRESHOLD = 0.5 

def play_sound():
    global currently_playing
    with lock:
        if not currently_playing and not pygame.mixer.music.get_busy():
            currently_playing = True
            pygame.mixer.music.play()

def stop_sound():
    global currently_playing
    with lock:
        if currently_playing and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            currently_playing = False

# Define model architectures
class EyeLocalizerRegression(neural.Module):
    def __init__(self, pretrained=False):
        super(EyeLocalizerRegression, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        self.features = neural.Sequential(*list(self.base_model.children())[:-1])
        self.regressor = neural.Sequential(
            neural.Linear(self.base_model.classifier.in_features, 256),
            neural.BatchNorm1d(256),
            neural.ReLU(),
            neural.Dropout(0.1),
            neural.Linear(256, 128),
            neural.BatchNorm1d(128),
            neural.ReLU(),
            neural.Dropout(0.1),
            neural.Linear(128, 4)
        )
    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.base_model.global_pool(x)
        x = x.view(x.size(0), -1)
        eye_coords = self.regressor(x)
        return eye_coords

class SimpleEyeClassifier(neural.Module):
    def __init__(self, num_classes=2):
        super(SimpleEyeClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = neural.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = neural.Linear(self.base_model.classifier.in_features, num_classes)
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
localizer = EyeLocalizerRegression(pretrained=False).to(device)
localizer.load_state_dict(torch.load("best_eye_localizer_model.pth", map_location=device))
localizer.eval()

classifier = SimpleEyeClassifier(num_classes=2).to(device)
classifier.load_state_dict(torch.load("eye_detector_model.pth", map_location=device))
classifier.eval()

# Haar Cascade for face detection only
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video capture
status = 0
cap = cv2.VideoCapture(0)
eyes_closed_time = 0

cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)

frame_count = 0
frame_interval = 1

while True:
    total_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    
    if frame_count % frame_interval != 0:
        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue 

    frame_h, frame_w = frame.shape[:2]
    debug_frame = frame.copy()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Drowsiness Detector", frame)
        cv2.imshow("Debug View", debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Use the first detected face (optional: you can improve by picking the biggest face)
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Crop face region
    face_roi = frame[y:y+h, x:x+w]
    resized_face = cv2.resize(face_roi, (128, 128))
    input_tensor = torch.from_numpy(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Localize eyes inside the cropped face
    with torch.no_grad():
        eye_coords = localizer(input_tensor).cpu().numpy()[0]

    # Convert normalized coordinates back to face ROI pixels
    left_eye_x = int(eye_coords[0] * w) + x
    left_eye_y = int(eye_coords[1] * h) + y
    right_eye_x = int(eye_coords[2] * w) + x
    right_eye_y = int(eye_coords[3] * h) + y

    # Draw eyes on debug frame
    cv2.circle(debug_frame, (left_eye_x, left_eye_y), 5, (0, 255, 255), -1)
    cv2.circle(debug_frame, (right_eye_x, right_eye_y), 5, (0, 255, 255), -1)

    # Define eye size
    eye_size = int(min(w, h) * 0.3)

    # Extract and process both eyes
    def extract_eye(eye_x, eye_y):
        top = max(0, eye_y - eye_size//2)
        bottom = min(frame_h, eye_y + eye_size//2)
        left = max(0, eye_x - eye_size//2)
        right = min(frame_w, eye_x + eye_size//2)
        eye_roi = frame[top:bottom, left:right]
        if eye_roi.size > 0:
            resized_eye = cv2.resize(eye_roi, (128, 128))
            eye_tensor = torch.from_numpy(cv2.cvtColor(resized_eye, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.
            eye_tensor = eye_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                output = classifier(eye_tensor)
                _, pred = torch.max(output, 1)
                is_open = pred.item() == 1
            return is_open, (left, top, right, bottom)
        return False, None

    left_eye_open, left_box = extract_eye(left_eye_x, left_eye_y)
    right_eye_open, right_box = extract_eye(right_eye_x, right_eye_y)

    # Draw eye status and boxes
    if left_box:
        color = (0, 255, 0) if left_eye_open else (0, 0, 255)
        cv2.rectangle(frame, (left_box[0], left_box[1]), (left_box[2], left_box[3]), color, 2)
    if right_box:
        color = (0, 255, 0) if right_eye_open else (0, 0, 255)
        cv2.rectangle(frame, (right_box[0], right_box[1]), (right_box[2], right_box[3]), color, 2)

    # Check eyes status
    eyes_open = left_eye_open or right_eye_open

    current_time = time.time()
    current_time = time.time()

    if eyes_open:
        last_eyes_open_time = current_time
        if not eyes_open_confirmed and (current_time - last_eyes_closed_time) > EYES_OPEN_THRESHOLD:
            # Confirm eyes are open after enough duration
            eyes_open_confirmed = True
            eyes_closed_confirmed = False
            threading.Thread(target=stop_sound, daemon=True).start()
            cv2.putText(frame, "AWAKE (confirmed)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "AWAKE (waiting confirm)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        last_eyes_closed_time = current_time
        if not eyes_closed_confirmed and (current_time - last_eyes_open_time) > EYES_CLOSED_THRESHOLD:
            # Confirm eyes are closed after enough duration
            eyes_closed_confirmed = True
            eyes_open_confirmed = False
            threading.Thread(target=play_sound, daemon=True).start()
            cv2.putText(frame, "DROWSY (confirmed)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            blink_time = current_time - last_eyes_open_time
            cv2.putText(frame, f"Eyes Closed (waiting confirm): {blink_time:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Drowsiness Detector", frame)
    cv2.imshow("Debug View", debug_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
