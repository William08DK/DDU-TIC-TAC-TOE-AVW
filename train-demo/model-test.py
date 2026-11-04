from ultralytics import YOLO
import os

# Mappen hvor dette script ligger – så virker stierne, uanset hvorfra du kører det
DIR = os.path.dirname(os.path.abspath(__file__))

# PRØV AT ÆNDRE: hedder din træningsmappe train-2, train-3 ..., så ret stien her
MODEL = os.path.join(DIR, "runs", "detect", "train", "weights", "best.pt")
IMAGE = os.path.join(
    DIR, "input-test.jpg"
)  # prøv fx et billede fra datasets/test/images
ROWS, COLS = 3, 5  # brættet har 3 rækker og 5 kolonner

if not os.path.exists(MODEL):
    print(f"Fandt ikke modellen: {MODEL}")
    print("Har du kørt model-train.py? Se i mappen runs/detect/, hvad din træning hedder.")
    raise SystemExit

model = YOLO(MODEL)
result = model(IMAGE)[0]  # ét billede ind = ét resultat ud

result.save(filename=os.path.join(DIR, "output-test-annotated.jpg"))
# result.show()

for box in result.boxes:
    name = result.names[int(box.cls)]  # klassens navn, fx "star"
    conf = float(box.conf)  # sikkerhed fra 0 til 1

    # Hjørner i pixels: (x1, y1) øverst til venstre, (x2, y2) nederst til højre
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # Midten normaliseret (0-1) – bruges til at finde feltet på brættet
    x, y, w, h = box.xywhn[0].tolist()
    row = int(y * ROWS) + 1  # +1 så vi tæller fra 1 i stedet for 0
    col = int(x * COLS) + 1

    print(
        f"{name:>8} {conf:.2f}  ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})  række {row}, kolonne {col}"
    )
