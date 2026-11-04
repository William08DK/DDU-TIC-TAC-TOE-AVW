from ultralytics import YOLO
import os

# Mappen hvor dette script ligger – så virker stierne, uanset hvorfra du kører det
DIR = os.path.dirname(os.path.abspath(__file__))

# PRØV AT ÆNDRE: værdierne herunder
MODEL = "yolo26n.pt"   # færdigtrænet model vi bygger videre på (n = nano, mindst og hurtigst)
DATA = os.path.join(DIR, "datasets", "data.yaml")   # beskrivelse af datasættet
EPOCHS = 30            # antal gennemløb af alle træningsbilleder
IMGSZ = 640            # billederne skaleres til 640 x 640 pixels
PATIENCE = 10          # stop tidligt, hvis modellen ikke er blevet bedre i 10 epochs

# Windows kræver denne linje, når træningen bruger flere processer
if __name__ == "__main__":
    model = YOLO(os.path.join(DIR, MODEL))

    # Træn modellen. Resultatet gemmes i train-demo/runs/detect/train
    model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        patience=PATIENCE,
        project=os.path.join(DIR, "runs", "detect"),
        name="train",
    )
