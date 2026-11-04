from ultralytics import YOLO
import os

# Mappen hvor dette script ligger – så virker stierne, uanset hvorfra du kører det
DIR = os.path.dirname(os.path.abspath(__file__))

# PRØV AT ÆNDRE: et andet billede i mappen input/
IMAGE = os.path.join(DIR, "input", "test.webp")

model = YOLO("yolo26n.pt")    # færdigtrænet model, der kender 80 hverdagsting
result = model(IMAGE)[0]      # ét billede ind = ét resultat ud

# Udskriv hvert objekt, YOLO har fundet
for box in result.boxes:
    name = result.names[int(box.cls)]   # klassens navn, fx "person"
    conf = float(box.conf)              # sikkerhed fra 0 til 1
    # Hjørner i pixels: (x1, y1) øverst til venstre, (x2, y2) nederst til højre
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{name:>8} {conf:.2f}  ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")

result.show()   # vis billedet med kasser
