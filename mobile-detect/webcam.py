# Mobile Detect – webcam
# Finder personer og mobiler med YOLO og sender status til serveren (server.py).
import sys
import cv2
import numpy as np
import requests
from ultralytics import YOLO

API_URL = "http://localhost:8080/update"

# PRØV AT ÆNDRE: kameraets nummer. 0 er som regel det indbyggede kamera, 1 et eksternt.
# Du kan også vælge kamera, når du starter programmet: uv run mobile-detect/webcam.py 1
CAMERA = 0
if len(sys.argv) > 1:
    CAMERA = int(sys.argv[1])

# PRØV AT ÆNDRE: hvor sikker YOLO skal være (0.0–1.0), før den tæller et fund med
CONFIDENCE = 0.5

# PRØV AT ÆNDRE: hvor mange billeder i træk, før status skifter (forhindrer blinken)
FRAMES_BEFORE_CHANGE = 5

# Hent den prætrænede YOLO26-model (downloades automatisk første gang)
model = YOLO("yolo26n.pt")

# YOLO's klasser har numre – vi slår dem op ud fra navnet
PERSON = "person"
PHONE = "cell phone"
class_ids = [i for i, name in model.names.items() if name in (PERSON, PHONE)]

# Åbn webcam
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    print(f"Kunne ikke åbne kamera {CAMERA}.")
    print("   Prøv et andet nummer, fx:  uv run mobile-detect/webcam.py 1")
    print("   Eller find dit kamera med:  uv run mobile-detect/find-camera.py")
    sys.exit(1)

phone_status = False  # den status, serveren kender
frames_with_change = 0  # hvor mange billeder i træk, der har vist noget andet


def send_status(status):
    """Send status til serveren. Kører serveren ikke, fortsætter vi bare."""
    try:
        requests.post(API_URL, json={"status": status}, timeout=1)
        print("Mobil fundet" if status else "⚪ Ingen mobil")
    except requests.RequestException:
        print("Kunne ikke kontakte serveren – kører server.py?")


send_status(phone_status)

while True:
    success, frame = cap.read()
    if not success:
        print("Kunne ikke læse billede fra kameraet.")
        break

    # Lad YOLO kigge efter personer og mobiler i billedet
    result = model(frame, classes=class_ids, conf=CONFIDENCE, verbose=False)[0]
    found = [model.names[int(c)] for c in result.boxes.cls]
    phone_seen = PHONE in found
    person_seen = PERSON in found

    # Skift først status, når YOLO har set det samme i flere billeder i træk
    if phone_seen != phone_status:
        frames_with_change += 1
    else:
        frames_with_change = 0

    if frames_with_change >= FRAMES_BEFORE_CHANGE:
        phone_status = phone_seen
        frames_with_change = 0
        send_status(phone_status)

    # Vælg farve (OpenCV bruger BGR: blå, grøn, rød)
    if phone_status:
        color = (0, 0, 255)  # rød
    elif person_seen:
        color = (0, 255, 0)  # grøn
    else:
        color = (255, 255, 255)  # hvid

    # Tegn YOLO's kasser og læg et farvet lag ovenpå
    annotated = result.plot()
    background = np.full_like(annotated, color, dtype=np.uint8)
    blended = cv2.addWeighted(annotated, 0.6, background, 0.4, 0)

    cv2.imshow("Mobile Detect (tryk q for at stoppe)", blended)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
