import cv2
import time
import threading
import pygame
import torch
import torch.nn as neural
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np

pygame.mixer.init()
pygame.mixer.music.load("Merry Go Round of Life.mp3")


lock = threading.Lock()
currently_playing = False

def play_sound():
    """Plays the sound only if it's not already playing."""
    global currently_playing
    with lock:  # Prevents simultaneous access to `currently_playing`
        if not currently_playing and not pygame.mixer.music.get_busy():
            currently_playing = True
            pygame.mixer.music.play()

def stop_sound():
    """Stops the sound playback only if it is currently playing."""
    global currently_playing
    with lock:
        if currently_playing and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            currently_playing = False

class EyeLocalizerRegression(neural.Module):
    def __init__(self, pretrained=True):
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
        self.base_model = timm.create_model('efficientnet_b0',pretrained=True)
        self.features = neural.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = neural.Linear(self.base_model.classifier.in_features, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

localizer = EyeLocalizerRegression(pretrained=False).to(device)
localizer.load_state_dict(torch.load("best_eye_localizer_model.pth", map_location=device))
localizer.eval()

classifier = SimpleEyeClassifier(num_classes=2).to(device)
classifier.load_state_dict(torch.load("eye_detector_model.pth", map_location=device))
classifier.eval() 

localizer_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

classifier_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

status = 0
cap = cv2.VideoCapture(0)
eyes_closed_time = 0

while(True):
    total_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Failed grab frame")
        break

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = localizer_transforms(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        eye_coords = localizer(input_tensor).cpu().numpy()[0]
    
    frame_h, frame_w = frame.shape[:2]
    
    left_eye_x, left_eye_y, right_eye_x, right_eye_y = eye_coords
    left_eye_x = int(left_eye_x * frame_w)
    left_eye_y = int(left_eye_y * frame_h)
    right_eye_x = int(right_eye_x * frame_w)
    right_eye_y = int(right_eye_y * frame_h)
    
    eye_size = int(min(frame_w, frame_h) * 0.1)
    
    left_eye_top = max(0, left_eye_y - eye_size//2)
    left_eye_bottom = min(frame_h, left_eye_y + eye_size//2)
    left_eye_left = max(0, left_eye_x - eye_size//2)
    left_eye_right = min(frame_w, left_eye_x + eye_size//2)
    
    right_eye_top = max(0, right_eye_y - eye_size//2)
    right_eye_bottom = min(frame_h, right_eye_y + eye_size//2)
    right_eye_left = max(0, right_eye_x - eye_size//2)
    right_eye_right = min(frame_w, right_eye_x + eye_size//2)
    
    left_eye_roi = frame[left_eye_top:left_eye_bottom, left_eye_left:left_eye_right]
    right_eye_roi = frame[right_eye_top:right_eye_bottom, right_eye_left:right_eye_right]
    
    left_eye_open = False
    right_eye_open = False
    
    if left_eye_roi.size > 0:
        # Convert to PIL Image for transformation
        left_eye_pil = Image.fromarray(cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2RGB))
        left_eye_tensor = classifier_transforms(left_eye_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            left_output = classifier(left_eye_tensor)
            _, left_pred = torch.max(left_output, 1)
            left_eye_open = bool(left_pred.item() == 1)  # Assuming class 1 is "open"
    
    if right_eye_roi.size > 0:
        # Convert to PIL Image for transformation
        right_eye_pil = Image.fromarray(cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2RGB))
        right_eye_tensor = classifier_transforms(right_eye_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            right_output = classifier(right_eye_tensor)
            _, right_pred = torch.max(right_output, 1)
            right_eye_open = bool(right_pred.item() == 1)  # Assuming class 1 is "open"
        
        
    eyes_open = left_eye_open or right_eye_open
    
    if left_eye_roi.size > 0:
        left_color = (0, 255, 0) if left_eye_open else (0, 0, 255)  # Green if open, red if closed
        cv2.rectangle(frame, 
                     (left_eye_left, left_eye_top), 
                     (left_eye_right, left_eye_bottom), 
                     left_color, 2)
    
    if right_eye_roi.size > 0:
        right_color = (0, 255, 0) if right_eye_open else (0, 0, 255)
        cv2.rectangle(frame, 
                     (right_eye_left, right_eye_top), 
                     (right_eye_right, right_eye_bottom), 
                     right_color, 2)

    cv2.putText(frame, 
                f"Left Eye: {'Open' if left_eye_open else 'Closed'}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2)
    
    cv2.putText(frame, 
                f"Right Eye: {'Open' if right_eye_open else 'Closed'}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2)

    if eyes_open and status != 1:
        threading.Thread(target=stop_sound, daemon=True).start()
        status = 1
        cv2.putText(frame, "AWAKE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif not eyes_open:
        if status != 2:
            eyes_closed_time = time.time()
        blink_time = total_time - eyes_closed_time
        
        # Show blink duration
        cv2.putText(frame, 
                    f"Eyes Closed: {blink_time:.1f}s", 
                    (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2)
        
        if blink_time > 0.5 and not currently_playing:
            threading.Thread(target=play_sound, daemon=True).start()
        status = 2
        
    cv2.imshow("Detector", frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows() 