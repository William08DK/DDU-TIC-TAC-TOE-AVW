# Eksempler – YOLO med en færdigtrænet model

Her prøver du YOLO, **før** du træner din egen model. Alle eksempler bruger den færdigtrænede model `yolo26n.pt`, som kender 80 hverdagsting (personer, biler, kopper, telefoner …). Se listen i [docs/coco-classes.json](../docs/coco-classes.json).

Første gang henter YOLO selv modellen (ca. 5 MB).

Kør alle kommandoer i VS Code-terminalen fra projektmappen (`C:\DDU\machine-learning-yolo`). Har du ikke installeret endnu, så følg [hoved-README'en](../README.md).

| Script | Input | Hvad sker der? | Stop |
|---|---|---|---|
| `image.py` | Ét billede | Finder objekter og udskriver koordinater | Luk billedet |
| `video.py` | En videofil | Finder og **tracker** objekter | Videoen slutter, eller `Ctrl+C` |
| `webcam.py` | Dit webcam | Finder og tracker objekter live | Tryk `q` |
| `workspace.py` | – | Viser forskellige måder at finde mapper på | – |

---

## image.py – find objekter i et billede

```powershell
uv run examples/image.py
```

Scriptet finder objekter i `input/test.webp`, viser billedet med kasser og udskriver én linje pr. objekt:

```
     bus 0.92  (154, 63) -> (377, 323)
  person 0.76  (306, 238) -> (355, 353)
```

- `bus` er **klassen** – hvad YOLO tror, det er.
- `0.92` er **sikkerheden** (*confidence*) fra 0 til 1.
- `(154, 63)` er kassens **øverste venstre hjørne** og `(377, 323)` det **nederste højre**, i pixels. Punktet `(0, 0)` er billedets øverste venstre hjørne, og y vokser nedad.

Det kaldes **detection** (objektgenkendelse): YOLO fortæller både *hvad* og *hvor*.

**Prøv at ændre:** `IMAGE` øverst i scriptet. Læg dit eget billede i `input/` og skriv filnavnet.

---

## video.py – tracking i en video

```powershell
uv run examples/video.py
```

Et vindue viser videoen `input/aerial-view.mp4`. Hvert objekt får en kasse **og et ID-nummer**, som følger objektet fra billede til billede. Det kaldes **tracking**.

Forskellen på de to linjer i scriptet:

```python
model(VIDEO, show=True)         # detection: hvert billede for sig, ingen ID'er
model.track(VIDEO, show=True)   # tracking: samme objekt beholder sit ID
```

Tracking bruger pakken `lap`, som er installeret med `uv sync`.

**Prøv at ændre:** Skift mellem de to linjer (sæt `#` foran den ene) og se forskellen. Prøv med din egen video i `input/` – ret `VIDEO` øverst i scriptet.

---

## webcam.py – live fra dit webcam

```powershell
uv run examples/webcam.py
```

Webcammet tænder, og YOLO tracker objekter live. Hold fx en kop, en telefon eller en bog op. **Tryk `q`** i webcam-vinduet for at stoppe.

Scriptet læser ét billede (*frame*) ad gangen fra kameraet i en løkke, lader YOLO finde objekter, tegner kasserne (`results[0].plot()`) og viser billedet med OpenCV (`cv2.imshow`).

**Prøv at ændre:** `CAMERA = 0` øverst i scriptet. `0` er det indbyggede kamera – prøv `1`, hvis du har et eksternt kamera.

---

## workspace.py – hvor kører mit script fra?

```powershell
uv run examples/workspace.py
```

Scriptet udskriver forskellige måder at finde mapper på. Kør det fra projektmappen, og kør det derefter med ▶-knappen i VS Code eller fra en anden mappe – så kan du se, at nogle af svarene ændrer sig.

Derfor bruger vores scripts denne linje:

```python
DIR = os.path.dirname(os.path.abspath(__file__))
```

Den giver altid mappen, hvor **scriptet** ligger. Så finder scriptet sine filer (fx `input/test.webp`), uanset hvorfra du kører det.

---

## Fejlfinding

**Fejl om, at filen ikke findes (`test.webp` eller `aerial-view.mp4`)**
Mappen `examples/input` mangler eller ligger forkert. Kør `git pull` eller tjek mappestrukturen i [hoved-README'en](../README.md).

**Webcam: sort billede, eller vinduet lukker med det samme**
- Windows: **Indstillinger → Beskyttelse af personlige oplysninger og sikkerhed → Kamera** – slå adgang til for apps og skrivebordsapps.
- Luk andre programmer, der bruger kameraet (Teams, Zoom, kamera-appen).
- Prøv `CAMERA = 1` i stedet for `0` øverst i `webcam.py`.
- **Mac:** Giv VS Code/Terminal adgang under *Systemindstillinger → Anonymitet og sikkerhed → Kamera*. Er der en iPhone i nærheden, kan Mac'en vælge iPhonens kamera – prøv et andet tal.

**`ModuleNotFoundError: No module named 'lap'`**
Kør `uv sync` i projektmappen og start scriptet med `uv run`, ikke `python`.

**Video og webcam hakker**
YOLO kører på computerens processor (CPU). På ældre computere er det normalt, at det går lidt langsomt.

---

**Næste skridt:** Træn din egen model i [train-demo](../train-demo/README.md).
