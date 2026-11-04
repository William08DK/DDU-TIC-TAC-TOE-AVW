# Find dit kamera
# Viser et billede fra hvert kamera, computeren kan finde, med kameraets nummer.
# Brug nummeret i webcam.py, fx:  uv run mobile-detect/webcam.py 1
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"  # skjul OpenCV's advarsler om kameraer, der ikke findes
import cv2

MAX_CAMERAS = 5  # hvor mange kameranumre vi prøver (0, 1, 2, ...)

found = []
for number in range(MAX_CAMERAS):
    cap = cv2.VideoCapture(number)
    success, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()

    if success:
        found.append(number)
        print(f"✅ Kamera {number} virker – tryk på en tast i billedvinduet for at gå videre")
        cv2.putText(frame, f"Kamera {number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Find kamera", frame)
        cv2.waitKey(0)

cv2.destroyAllWindows()

if found:
    print(f"\nFundne kameraer: {found}")
else:
    print("❌ Ingen kameraer fundet. Tjek at kameraet er tilsluttet, og at programmet har adgang til det.")
