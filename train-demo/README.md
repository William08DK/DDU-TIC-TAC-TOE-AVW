# Træn din egen YOLO-model – demo med brætspillet

I denne demo træner vi YOLO til at genkende brikkerne på vores brætspil. Brættet har 3 rækker × 5 felter, og brikkerne har seks forskellige symboler.

> **Du er færdig, når** du har trænet en model, testet den på et billede og kan forklare, hvad tallene i terminalen betyder.

Du skal have gennemført installationen i [hoved-README'en](../README.md) (`uv sync` og eksemplerne virker).

---

## 1. Hvad er en *detect*-opgave?

YOLO kan løse flere slags opgaver. I denne demo bruger vi **detect** (objektgenkendelse):

| Opgave | Hvad får du ud? | Eksempel |
|---|---|---|
| *classify* | Én klasse for hele billedet | Teachable Machine: "det er en kat" |
| **detect** | For **hvert** objekt: klasse + kasse (*bounding box*) + sikkerhed | "star i (172, 82)–(262, 202), 86 % sikker" |
| *segment* | Som detect, men med objektets præcise omrids | Bruges ikke her |

Detect giver os altså både **hvad** der er på billedet og **hvor** det er. Det er det, vi skal bruge for at vide, hvilken brik der ligger på hvilket felt.

Vi starter fra den færdigtrænede model `yolo26n.pt` (trænet på 80 hverdagsting som personer og biler) og lærer den vores egne seks klasser. Det kaldes *transfer learning* – det går meget hurtigere end at starte fra nul.

---

## 2. Datasættet

```
train-demo/
  model-train.py         ← træner modellen
  model-test.py          ← finder brikker i input-test.jpg og skriver koordinater
  model-test2.py         ← samme, men skriver brættet som et gitter af bogstaver
  input-test.jpg         ← testbillede
  output-test-annotated.jpg  ← testbilledet med kasser (laves af model-test.py)
  datasets/
    data.yaml        ← fortæller YOLO, hvor billederne er, og hvad klasserne hedder
    train/           ← 44 billeder, som modellen lærer af
      images/
      labels/
    valid/           ← 10 billeder, som bruges til at måle modellen undervejs
      images/
      labels/
    test/            ← 9 billeder, som modellen aldrig har set
      images/
      labels/
```

**Hvorfor tre mapper?** Modellen må ikke blive testet på de billeder, den har lært af – så ville vi ikke vide, om den har *forstået* brikkerne eller bare *husket* billederne (det kaldes *overfitting*).

### data.yaml

```yaml
# Stierne er relative til mappen, hvor denne fil ligger
train: train/images
val: valid/images
test: test/images

# Antal klasser og deres navne. Rækkefølgen bestemmer klassenummeret (circle = 0, maltese = 1 ...)
nc: 6
names: ['circle', 'maltese', 'nothing', 'square', 'star', 'temple']
```

`nc` er antallet af klasser. Rækkefølgen i `names` er vigtig: `circle` er klasse **0**, `maltese` er klasse **1** osv.

### Label-filer

Hvert billede har en `.txt`-fil med **samme navn** i `labels/`. Hver linje er én brik:

```
5 0.51875 0.503125 0.13984375 0.1796875
```

| 5 | 0.51875 | 0.503125 | 0.1398… | 0.1796… |
|---|---|---|---|---|
| klasse (temple) | x for midten | y for midten | bredde | højde |

Tallene er **normaliserede** – dvs. en andel af billedets bredde/højde (0 til 1). `0.5, 0.5` er midt i billedet, uanset hvor mange pixels billedet har.

---

## 3. Træn modellen

Kør i VS Code-terminalen fra projektmappen (`C:\DDU\machine-learning-yolo`):

```powershell
uv run train-demo/model-train.py
```

(Du kan også åbne `model-train.py` og trykke ▶.)

Øverst i `model-train.py` står de værdier, du kan ændre:

```python
MODEL = "yolo26n.pt"   # færdigtrænet model vi bygger videre på
EPOCHS = 30            # antal gennemløb af alle træningsbilleder
IMGSZ = 640            # billederne skaleres til 640 x 640 pixels
PATIENCE = 10          # stop tidligt, hvis modellen ikke er blevet bedre i 10 epochs
```

**Det tager tid.** På en ny Mac tog 100 epochs ca. 9 minutter. På en Windows-laptop uden grafikkort tager det meget længere – sæt evt. `EPOCHS = 10` første gang for at se, at det virker.

Undervejs skriver YOLO en linje pr. epoch. Hold øje med:

| Begreb | Betydning |
|---|---|
| `box_loss`, `cls_loss` | Fejl på kassernes placering og på klassen. Skal **falde**. |
| `P` (*precision*) | Når modellen siger "star", hvor ofte har den så ret? |
| `R` (*recall*) | Hvor mange af de rigtige brikker fandt modellen? |
| `mAP50` | Samlet karakter fra 0 til 1. Over 0.8 er godt for os. |

Til sammenligning: med 10 epochs fik vi `mAP50 = 0.49`, med 100 epochs `mAP50 = 0.83`.

### Hvor ligger resultatet?

```
train-demo/runs/detect/train/
  weights/best.pt          ← DIN MODEL (den bedste epoch)
  weights/last.pt          ← modellen efter sidste epoch
  results.png              ← grafer over loss og mAP pr. epoch
  confusion_matrix.png     ← hvilke klasser bliver forvekslet?
  val_batch0_pred.jpg      ← modellens gæt på valideringsbillederne
```

Træner du igen, kommer resultatet i en ny mappe: `train-2`, `train-3` … Den gamle bliver ikke overskrevet.

---

## 4. Test modellen

```powershell
uv run train-demo/model-test.py
```

Scriptet indlæser `runs/detect/train/weights/best.pt`, finder brikker i `input-test.jpg` og gemmer billedet med kasser som `output-test-annotated.jpg`. Åbn filen for at se resultatet.

> Vil du have billedet vist med det samme, så fjern `#` foran `# result.show()` i scriptet.

> Hedder din træningsmappe `train-2` eller lignende, så ret `MODEL` øverst i `model-test.py`.

### Koordinaterne

I terminalen får du én linje pr. brik (eksempel):

```
    star 0.99  (172, 83) -> (262, 200)  række 1, kolonne 2
 maltese 0.98  (285, 439) -> (374, 558)  række 3, kolonne 3
  temple 0.95  (64, 84) -> (155, 199)  række 1, kolonne 1
```

- `0.95` er modellens sikkerhed (*confidence*), 95 %.
- `(64, 84)` er kassens **øverste venstre hjørne** og `(155, 199)` det **nederste højre** – i pixels. Det hedder `xyxy` i Ultralytics.
- Punktet `(0, 0)` er billedets **øverste venstre hjørne**. y vokser **nedad**.

Ultralytics kan give koordinaterne i flere formater:

| Kode | Giver |
|---|---|
| `box.xyxy` | hjørner i pixels |
| `box.xywh` | midte + bredde/højde i pixels |
| `box.xywhn` | midte + bredde/højde, normaliseret (0–1) – samme format som label-filerne |

### Fra koordinat til felt på brættet

Scriptet bruger midten fra `box.xywhn` til at finde feltet. Brættet fylder (næsten) hele billedet og har 3 rækker og 5 kolonner:

```python
x, y, w, h = box.xywhn[0].tolist()
row = int(y * ROWS) + 1   # y = 0.50 → int(1.5) + 1 = række 2
col = int(x * COLS) + 1   # x = 0.85 → int(4.25) + 1 = kolonne 5
```

Det virker kun, når kameraet ser brættet lige ovenfra og brættet fylder billedet. Flytter du kameraet, skal du justere beregningen.

### Brættet som tekst – `model-test2.py`

```powershell
uv run train-demo/model-test2.py
```

`model-test2.py` bruger samme model og samme billede, men skriver hele brættet ud som et gitter med ét bogstav pr. felt:

```
T S M - S
S T T M C
- - M - -
```

| Bogstav | C | M | F | S | T | - |
|---|---|---|---|---|---|---|
| Klasse | circle | maltese | square (firkant) | star | temple | nothing / tomt felt |

Sådan virker det:

- Brættet gemmes i en **dictionary**: nøglen er `(række, kolonne)` og værdien er `(navn, sikkerhed)`. Felter uden brik kommer ikke med i dictionary'en og bliver skrevet som `-`.
- Her tæller række og kolonne fra **0** (ikke fra 1 som i `model-test.py`), fordi de bruges direkte i `range(ROWS)` og `range(COLS)`.
- `min(..., ROWS - 1)` sørger for, at en brik med midten helt ude på kanten (1.0) ikke havner i en række, der ikke findes.
- Finder modellen **to brikker i samme felt**, bliver den med højest sikkerhed (*confidence*) valgt.

Det er et godt udgangspunkt, hvis dit eget spil skal vide, hvad der ligger på brættet – fx for at afgøre, hvem der har vundet.

---

## 5. Videre: træn på dine egne data

Når du vil træne på dit eget spil, skal du gøre det samme som i demoen:

1. **Tag billeder** – gerne 50+ med forskelligt lys, vinkler og placeringer.
2. **Annotér** – tegn kasser og vælg klasse i [MakeSense](https://www.makesense.ai/) (eller Roboflow). Eksportér i **YOLO-format**.
3. **Fordel** billederne i `train/`, `valid/` og `test/` (fx 70 / 20 / 10 %). Hvert billede skal have sin label-fil med samme navn.
4. **Ret `data.yaml`**: `nc` og `names` skal passe til dine klasser – i samme rækkefølge som i annoteringsværktøjet.
5. **Træn** med `model-train.py` og **test** med `model-test.py` / `model-test2.py` på et billede, modellen ikke har set. Husk at rette `LETTERS` i `model-test2.py`, så den passer til dine klasser.

---

## Hvad kan du ændre?

| Hvor | Værdi | Effekt |
|---|---|---|
| `model-train.py` | `EPOCHS` | Færre = hurtigere, men dårligere model |
| `model-train.py` | `IMGSZ` | Mindre (fx 320) = hurtigere, men små detaljer forsvinder |
| `model-train.py` | `PATIENCE` | Hvor mange epochs uden forbedring, før træningen stopper af sig selv |
| `model-train.py` | `MODEL` | `n` = nano (mindst/hurtigst). `yolo26s.pt` er større og mere præcis, men langsommere |
| `model-test.py` | `MODEL` | Stien til den model, du vil teste (fx `train-2`) |
| `model-test.py` | `IMAGE` | Prøv med et andet billede, fx fra `datasets/test/images/` |
| `model-test2.py` | `MODEL`, `IMAGE` | Som i `model-test.py` |
| `model-test2.py` | `ROWS, COLS` | Brættets størrelse – ret dem, hvis dit bræt har et andet antal felter |
| `model-test2.py` | `LETTERS` | Hvilket bogstav hver klasse skrives som |

---

## Fejlfinding

**`Fandt ikke modellen: ...best.pt`**
Du har ikke trænet endnu, eller træningen ligger i `train-2` osv. Se i `train-demo/runs/detect/` og ret `MODEL` i `model-test.py` (og `model-test2.py`). `model-test2.py` giver i stedet en længere fejl med `FileNotFoundError` – årsagen er den samme.

**Der står `?` i gitteret fra `model-test2.py`**
Modellen har fundet en klasse, som ikke står i `LETTERS`. Tilføj den i `LETTERS`.

**`Dataset ... does not exist` / `images not found`**
Mappen `train-demo/datasets` mangler eller er flyttet. Kør `git pull`, og tjek at `data.yaml` ligger i `datasets/` sammen med `train/`, `valid/` og `test/`.

**Træningen er meget langsom**
Det er normalt uden grafikkort. Sæt `EPOCHS` ned (fx 10) eller `IMGSZ = 320`.

**`RuntimeError: An attempt has been made to start a new process …`** (Windows)
Linjen `if __name__ == "__main__":` i `model-train.py` skal stå der – slet den ikke. Hjælper det ikke, så tilføj `workers=0` i `model.train(...)`.

**Modellen finder ingenting på mine egne billeder**
Tjek at label-filerne har samme navn som billederne, og at `nc`/`names` i `data.yaml` passer. Se `runs/detect/train/train_batch0.jpg` – her kan du se, om dine kasser sidder rigtigt.
