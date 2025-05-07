import cv2
import time
import threading
import pygame

pygame.mixer.init()
pygame.mixer.music.load("Merry Go Round of Life.mp3")


lock = threading.Lock()
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
    

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if eye_cascade.empty():
    print("Error loading eye cascade. Check the file path.")

status = 0
cap = cv2.VideoCapture(0)
currently_playing = False
eyes_closed_time = 0

while(True):
    total_time = time.time()
    
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


    if len(eyes) > 0 and status != 1:
        threading.Thread(target=stop_sound, daemon=True).start()
        status = 1
    elif len(eyes) == 0:
        if(status != 2):
            eyes_closed_time = time.time()
        blink_time = total_time - eyes_closed_time
        if blink_time > 0.5 and currently_playing == False:
            threading.Thread(target=play_sound, daemon=True).start()
        status = 2
    else:
        print("AWAKE")

        
    cv2.imshow("Detector", frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows() 