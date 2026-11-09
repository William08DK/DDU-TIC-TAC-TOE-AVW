"""Træn YOLO på brikkerne i lærerens datasæt."""
from pathlib import Path
from ultralytics import YOLO

DIR = Path(__file__).resolve().parent
EPOCHS = 30

if __name__ == "__main__":
    model = YOLO(str(DIR / "yolo26n.pt"))
    model.train(
        data=str(DIR / "datasets/data.yaml"),
        epochs=EPOCHS,
        imgsz=640,
        patience=10,
        project=str(DIR / "runs/detect"),
        name="train",
    )
