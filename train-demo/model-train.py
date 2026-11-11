"""Træn YOLO på vores egne kryds- og bollebrikker."""
from pathlib import Path
from ultralytics import YOLO

DIR = Path(__file__).resolve().parent
EPOCHS = 50
DATA = DIR / "egne-brikker/data.yaml"

if __name__ == "__main__":
    if not DATA.is_file():
        raise SystemExit("Datasættet mangler. Kør først: uv run train-demo/prepare_data.py")
    model = YOLO(str(DIR / "yolo26n.pt"))
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=640,
        patience=10,
        workers=0,
        project=str(DIR / "runs/detect"),
        name="egne-brikker",
    )
