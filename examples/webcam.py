from ultralytics import YOLO
import cv2

# PRØV AT ÆNDRE: 0 er det indbyggede kamera – prøv 1, hvis du har et eksternt kamera
CAMERA = 0

model = YOLO("yolo26n.pt")    # færdigtrænet model, der kender 80 hverdagsting
cap = cv2.VideoCapture(CAMERA)

if not cap.isOpened():
    print(f"Kunne ikke åbne kamera {CAMERA}. Prøv et andet tal i CAMERA øverst i scriptet,")
    print("og tjek at andre programmer (Teams, Zoom ...) ikke bruger kameraet.")
    raise SystemExit

print("Tryk q i webcam-vinduet for at stoppe.")

# Løkken læser ét billede (frame) ad gangen fra kameraet
while True:
    success, frame = cap.read()
    if not success:
        print("Fik ikke noget billede fra kameraet.")
        break

    # Tracking: persist=True husker objekterne fra forrige billede, så de beholder deres ID
    results = model.track(frame, persist=True, verbose=False)

    # Tegn kasser, navne og ID'er på billedet og vis det
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO webcam", annotated_frame)

    # Stop, når der trykkes på q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Luk kameraet og vinduet igen
cap.release()
cv2.destroyAllWindows()
