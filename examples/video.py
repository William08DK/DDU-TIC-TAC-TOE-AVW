from ultralytics import YOLO
import os

# Mappen hvor dette script ligger – så virker stierne, uanset hvorfra du kører det
DIR = os.path.dirname(os.path.abspath(__file__))

# PRØV AT ÆNDRE: din egen video i mappen input/
VIDEO = os.path.join(DIR, "input", "aerial-view.mp4")

model = YOLO("yolo26n.pt")    # færdigtrænet model, der kender 80 hverdagsting

# PRØV AT ÆNDRE: sæt # foran tracking-linjen og fjern # foran detection-linjen

# Detection: hvert billede i videoen behandles for sig – ingen ID'er
# model(VIDEO, show=True)

# Tracking: hvert objekt får et ID, som følger det fra billede til billede
model.track(VIDEO, show=True)
