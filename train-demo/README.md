# Træning af brikgenkendelse

Datasættet fra læreren er bevaret med klasserne `circle`, `maltese`, `nothing`, `square`, `star` og `temple`. Spillet bruger som standard `maltese` som X og `circle` som O.

Kør fra projektets hovedmappe:

```powershell
uv run train-demo/model-train.py
```

Træningen bruger `datasets/data.yaml` og gemmer modellen under `train-demo/runs/detect/train/weights/best.pt`. Ved senere træninger kan Ultralytics vælge en ny mappe. Brug den udskrevne sti med `main.py --model STI`.

`EPOCHS` i scriptet bestemmer antal træningsrunder. Basismodellen `yolo26n.pt` hentes automatisk, hvis den mangler. Det kræver internet første gang.

De oprindelige billeder viser et 3 × 5-bræt. Test derfor genkendelsen på jeres faktiske 3 × 3-bræt. Til almindelige X/O-brikker skal I annotere nye billeder, opdatere klasser og stier i `datasets/data.yaml`, træne igen og vælge de nye klassenavne med `--x-class` og `--o-class`.
