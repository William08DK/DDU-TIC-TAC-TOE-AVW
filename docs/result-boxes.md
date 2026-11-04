# `result.boxes` – kasserne fra YOLO

Når YOLO har kigget på et billede, får du et **resultat** tilbage. I `result.boxes` ligger alle de objekter, modellen har fundet – én kasse (*bounding box*) pr. objekt.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")           # eller din egen best.pt
result = model("billede.jpg")[0]     # ét billede ind = ét resultat ud
boxes = result.boxes
```

Eksemplerne herunder er kørt på `train-demo/input-test.jpg` med modellen fra `train-demo` (Ultralytics 8.4).

---

## Attributter

| Attribut | Hvad er det? | Eksempel (én kasse) |
|---|---|---|
| `cls` | Klassenummeret. Slå navnet op i `result.names` | `0` → `"circle"` |
| `conf` | Sikkerhed (*confidence*) fra 0 til 1 | `0.96` |
| `xyxy` | Hjørner i pixels: øverst til venstre og nederst til højre | `[499, 258, 593, 380]` |
| `xywh` | Midte + bredde/højde i pixels | `[546, 319, 94, 123]` |
| `xyxyn` | Som `xyxy`, men normaliseret (0–1) | `[0.78, 0.40, 0.93, 0.59]` |
| `xywhn` | Som `xywh`, men normaliseret (0–1) – samme format som label-filerne | `[0.85, 0.50, 0.15, 0.19]` |
| `id` | Trackingnummer – kun ved `model.track(...)`, ellers `None` | `1` |
| `is_track` | `True`, hvis kasserne kommer fra tracking | `False` |
| `orig_shape` | Billedets størrelse som `(højde, bredde)` | `(640, 640)` |
| `data` | Alle tal på én gang: x1, y1, x2, y2, (id), conf, cls | – |

Husk: `(0, 0)` er billedets **øverste venstre hjørne**, og y vokser **nedad**.

**Normaliseret** betyder, at tallet er en andel af billedets bredde/højde. `x = 0.5` er altid midt i billedet, uanset hvor mange pixels det har. Det er praktisk, når du vil finde et felt på et bræt.

## Metoder og nyttige funktioner

| Kode | Hvad gør den? |
|---|---|
| `len(result.boxes)` | Antal fundne objekter |
| `result.boxes[0]` | Den første kasse |
| `result.boxes[result.boxes.conf > 0.9]` | Kun kasserne med sikkerhed over 0.9 |
| `.tolist()` | Laver tallene om til en almindelig Python-liste |
| `int(box.cls)`, `float(box.conf)` | Laver ét tal om til `int`/`float` |
| `.cpu()`, `.numpy()` | Flytter data til CPU / laver dem om til NumPy (fx til OpenCV) |
| `result.names` | Dictionary fra klassenummer til navn, fx `{0: 'circle', 1: 'maltese', ...}` |
| `result.summary()` | Alle kasser som en liste af dictionaries |

### Hvorfor `[0]` og `.tolist()`?

YOLO gemmer tallene som *tensors* (PyTorch' version af lister). For én kasse er `box.xyxy` en tabel med **én række**:

```python
print(box.xyxy)       # tensor([[499.2399, 257.8614, 592.9515, 380.3850]])
print(box.xyxy[0])    # tensor([499.2399, 257.8614, 592.9515, 380.3850])
x1, y1, x2, y2 = box.xyxy[0].tolist()   # fire almindelige tal
```

---

## Eksempler

### 1. Hvor mange objekter fandt modellen?

```python
print(len(result.boxes))
```
```
10
```

### 2. Gennemløb alle kasser

Det mest almindelige: navn, sikkerhed og placering for hver kasse.

```python
for box in result.boxes:
    name = result.names[int(box.cls)]   # klassens navn, fx "circle"
    conf = float(box.conf)              # sikkerhed fra 0 til 1
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{name} {conf:.2f}  ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
```
```
circle 0.96  (499, 258) -> (593, 380)
temple 0.93  (174, 259) -> (263, 381)
temple 0.93  (62, 81) -> (155, 203)
...
```

### 3. Find midten af en kasse

```python
box = result.boxes[0]
x, y, w, h = box.xywh[0].tolist()
print(f"Midte: ({x:.0f}, {y:.0f}), bredde {w:.0f}, højde {h:.0f}")
```
```
Midte: (546, 319), bredde 94, højde 123
```

### 4. Kun de sikre fund

```python
sikre = result.boxes[result.boxes.conf > 0.9]
print(len(sikre), "af", len(result.boxes), "brikker er over 90 % sikre")
```
```
6 af 10 brikker er over 90 % sikre
```

Du kan også bede YOLO om at sortere de usikre fra med det samme: `model("billede.jpg", conf=0.9)`.

### 5. Tæl objekter pr. klasse

```python
from collections import Counter

navne = [result.names[int(c)] for c in result.boxes.cls]
print(Counter(navne))
```
```
Counter({'temple': 3, 'maltese': 3, 'star': 3, 'circle': 1})
```

### 6. Alle værdier som lister

```python
print(result.boxes.cls.tolist())    # klassenumre
print(result.boxes.conf.tolist())   # sikkerheder
```
```
[0.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0]
[0.9629184007644653, 0.9283542633056641, 0.927493155002594, ...]
```

### 7. Fra koordinat til felt på brættet

Med de normaliserede tal kan du finde række og kolonne – her for et bræt med 3 rækker og 5 kolonner (se også `train-demo/model-test.py` og `model-test2.py`):

```python
ROWS, COLS = 3, 5
x, y, w, h = box.xywhn[0].tolist()
row = min(int(y * ROWS), ROWS - 1)   # 0, 1 eller 2
col = min(int(x * COLS), COLS - 1)   # 0 til 4
```

### 8. Det hele som dictionaries

```python
for d in result.summary():
    print(d)
```
```
{'name': 'circle', 'class': 0, 'confidence': 0.96292, 'box': {'x1': 499.23993, 'y1': 257.86139, 'x2': 592.95148, 'y2': 380.38501}}
...
```

### 9. Tracking – følg det samme objekt fra billede til billede

Bruger du `model.track(...)` i stedet for `model(...)`, får hver kasse et `id`, som bliver ved med at høre til det samme objekt i en video eller på webcam.

```python
result = model.track("billede.jpg")[0]
print(result.boxes.is_track)        # True
print(result.boxes.id.tolist())     # [1.0, 2.0, 3.0, ...]
```

Tracking kræver pakken `lap` (er med, når du har kørt `uv sync`). Uden tracking er `box.id` `None`.

---

## Læs mere

- Ultralytics-dokumentation for `Boxes`: https://docs.ultralytics.com/modes/predict/#boxes
- Alle egenskaber ved resultatet: https://docs.ultralytics.com/modes/predict/#working-with-results
