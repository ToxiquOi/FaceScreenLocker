import cv2
import time
import ctypes

frontface_default = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
profileface = cv2.data.haarcascades + "haarcascade_profileface.xml"

print(profileface)
print(frontface_default)

front_hog = cv2.CascadeClassifier(frontface_default)
profile_hog = cv2.CascadeClassifier(profileface)

cv2.startWindowThread()
cap = cv2.VideoCapture(0)

last_detect_time=0
lock_required=False

while (True):
    # capture image par image
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes = front_hog.detectMultiScale(gray, 1.1, 4)
    profil_boxes = profile_hog.detectMultiScale(gray, 1.1, 4)

    if len(boxes) > 0 or len(profil_boxes) > 0:
        last_detect_time = time.perf_counter()
        lock_required = False

    else:
        print(time.perf_counter() - last_detect_time)
        if time.perf_counter() - last_detect_time > 30:
            lock_required = True

    time.sleep(0.25)

    if lock_required:
        ctypes.windll.user32.LockWorkStation()
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
