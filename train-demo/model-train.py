"""Træn YOLO på vores egne kryds- og bollebrikker."""
from pathlib import Path
from ultralytics import YOLO

# Find træningsmappen ud fra denne fil, ikke ud fra terminalens aktuelle mappe.
DIR = Path(__file__).resolve().parent
# Én epoch er én træningsrunde gennem træningsbillederne.
EPOCHS = 50
# YAML-filen fortæller, hvor billeder/labels er, og hvad de to klasser hedder.
DATA = DIR / "nye-brikker/data.yaml"

# Denne kontrol er vigtig på Windows, så filen ikke starter træning ved import.
if __name__ == "__main__":
    if not DATA.is_file():
        raise SystemExit("Datasættet mangler. Kør først: uv run train-demo/prepare_data.py")
    # Start med en basismodel, som allerede har lært generelle billedmønstre.
    # Under træningen tilpasses den til vores egne labels: kryds og bolle.
    model = YOLO(str(DIR / "yolo26n.pt"))
    # Træningen sammenligner modellens gæt med vores håndtegnede markeringer.
    # Den justerer sine tal (modelvægte), så gættene gradvist bliver bedre.
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=640,  # Billederne tilpasses en analyseopløsning på 640 under træning.
        patience=10,  # Stop tidligt efter 10 runder uden forbedring af valideringsresultatet.
        workers=0,  # Indlæs data i hovedprocessen; det holder Windows-opsætningen enkel.
        project=str(DIR / "runs/detect"),
        name="nye-brikker",  # Resultater og weights/best.pt gemmes under denne mappe.
    )
