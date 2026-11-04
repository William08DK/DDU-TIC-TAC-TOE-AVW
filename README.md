# Machine Learning med YOLO

I dette forløb bruger vi **YOLO** (fra Ultralytics) til at genkende objekter i billeder, video og live fra dit webcam.

Denne guide viser, hvordan du får YOLO op at køre på din egen computer.

> **Du er færdig, når** du kan køre alle tre eksempler: `image.py`, `video.py` og `webcam.py`.

Guiden er skrevet til **Windows**. Hvor Mac er anderledes, står det i en note.

---

## Del 1. Installation og opsætning

### Trin 1 – Installér programmer

Du skal bruge tre programmer:

| Program | Hvad bruger vi det til? | Download |
|---|---|---|
| **VS Code** | Editor, hvor vi skriver og kører kode | https://code.visualstudio.com/ |
| **git** | Henter projektet fra GitHub | https://git-scm.com/install/ |
| **uv** | Installerer Python og de biblioteker, vi skal bruge | https://docs.astral.sh/uv/ |

Hvis du allerede har programmerne, behøver du ikke at installere dem igen. Du behøver **ikke** installere Python selv – det klarer uv. 
Har du allerede Python, gør det ikke noget.

**Installér VS Code og git** fra linkene ovenfor. Brug standardindstillingerne i installationsprogrammet (bare tryk *Next*).

**Installér uv:** Åbn *PowerShell* (tryk på Windows-tasten, skriv `powershell`, tryk Enter) og kør:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> **Mac:** Åbn *Terminal* og kør `curl -LsSf https://astral.sh/uv/install.sh | sh`
> git findes ofte allerede på Mac – ellers bliver du spurgt, om du vil installere det, første gang du skriver `git`.

**Luk alle vinduer med PowerShell og VS Code**, når du er færdig. Programmerne bliver først fundet, når terminalen startes forfra.

### Trin 2 – Tjek at det virker

Åbn PowerShell igen og skriv:

```powershell
git --version
uv --version
```

Begge kommandoer skal svare med et versionsnummer. Får du en fejl, så se [Fejlfinding](#fejlfinding).

### Trin 3 – Opsæt VS Code

1. Åbn VS Code.
2. Klik på *Extensions* i venstre side (ikonet med fire firkanter) eller tryk `Ctrl+Shift+X`.
3. Søg efter **Python** og installér udvidelsen fra **Microsoft**.

### Trin 4 – Hent projektet fra GitHub

Vi anbefaler, at du laver en **tom mappe** til faget, fx `C:\DDU`.

> Læg **ikke** mappen i OneDrive (fx *Dokumenter* eller *Skrivebord*, hvis de synkroniseres). OneDrive kan give problemer med de mange filer, som installeres.

I PowerShell:

```powershell
mkdir C:\DDU
cd C:\DDU
git clone https://github.com/slotshaven-ddu/machine-learning-yolo
```

Du kan også bruge kommandoen i VS Code (CTRL-SHIFT-P, skriv `git clone`).

Nu har du mappen `C:\DDU\machine-learning-yolo` med alle filerne.

> **Mac:** fx `mkdir ~/DDU`, `cd ~/DDU` og derefter samme `git clone`-kommando.

<details>
<summary><b>Alternativ: hent filerne uden git</b></summary>

Virker git ikke, kan du hente projektet som en zip-fil:

1. Gå til https://github.com/slotshaven-ddu/machine-learning-yolo
2. Klik på den grønne knap **Code** → **Download ZIP**.
3. Pak zip-filen ud i `C:\DDU` og omdøb mappen fra `machine-learning-yolo-main` til `machine-learning-yolo`.

Du kan også downloade filerne enkeltvis fra GitHub. Så skal du selv lave den samme mappestruktur – især skal `examples/input/` med `test.webp` og `aerial-view.mp4` være med:

```
machine-learning-yolo/
  pyproject.toml
  uv.lock
  .python-version
  examples/
    image.py
    video.py
    webcam.py
    input/
      test.webp
      aerial-view.mp4
```

Husk også filerne `pyproject.toml`, `uv.lock` og `.python-version` i hovedmappen – uden dem ved uv ikke, hvad der skal installeres.

Ulempen ved at hente filerne uden git er, at du ikke kan hente opdateringer med `git pull`.
</details>

### Trin 5 – Åbn projektet i VS Code

1. I VS Code: **File → Open Folder…** og vælg `C:\DDU\machine-learning-yolo`.
2. Svar **Yes, I trust the authors**, hvis VS Code spørger.
3. Åbn en terminal i VS Code: **Terminal → New Terminal**.

Terminalen skal være **PowerShell** (det er standard på Windows).
Tjek, at terminalen står i projektmappen – der skal stå `C:\DDU\machine-learning-yolo>` før markøren.

### Trin 6 – Installér Python-biblioteker

Kør denne kommando i VS Code-terminalen:

```powershell
uv sync
```

uv læser projektets filer (`pyproject.toml` og `uv.lock`) og installerer præcis de samme versioner som alle andre i klassen – i mappen `.venv`:

- **Python 3.13** (hvis du ikke har det i forvejen)
- **ultralytics** – selve YOLO
- **opencv-python** – billeder, video og webcam
- **lap** – bruges, når YOLO skal følge (tracke) objekter i video

Det tager nogle minutter første gang, fordi der skal hentes ret meget (bl.a. PyTorch). Vent til terminalen er klar igen.

> Glemmer du dette trin, gør `uv run` det automatisk første gang.

### Trin 7 – Kør eksemplerne

Kør eksemplerne ét ad gangen fra VS Code-terminalen. Første gang henter YOLO automatisk modellen `yolo26n.pt` (ca. 5 MB).

**1. Billede**

```powershell
uv run examples/image.py
```

Billedet åbner med kasser omkring de objekter, YOLO har fundet (personer, busser, cykler …). I terminalen kan du se, hvad der er fundet, hvor sikker YOLO er, og hvor objektet er i billedet (kassens hjørner i pixels).

**2. Video**

```powershell
uv run examples/video.py
```

Et vindue viser videoen, hvor hvert objekt får en kasse og et ID-nummer. Det er *tracking*. Videoen stopper af sig selv – eller tryk `Ctrl+C` i terminalen.

**3. Webcam**

```powershell
uv run examples/webcam.py
```

Dit webcam tænder, og YOLO genkender ting live. Prøv at holde en kop, en telefon eller en bog op. **Tryk `q`** i webcam-vinduet for at stoppe.

> **Tip:** Du kan også åbne en `.py`-fil og trykke på ▶ (*Run Python File*) øverst til højre i VS Code. Vælg i så fald Python fra mappen `.venv`, hvis VS Code spørger om fortolker (*interpreter*).

### Tjekliste – er du klar?

- [ ] `git --version` og `uv --version` virker
- [ ] Projektet ligger i en mappe for sig selv, fx `C:\DDU\machine-learning-yolo`
- [ ] `uv sync` er kørt uden fejl
- [ ] `uv run examples/image.py` viser et billede med kasser
- [ ] `uv run examples/video.py` viser video med kasser og ID'er
- [ ] `uv run examples/webcam.py` genkender ting fra dit webcam

Så er du klar til at træne din egen model!

---

## Fejlfinding

**`uv` eller `git` "is not recognized as the name of a cmdlet …"**
Programmet er installeret, men terminalen kender det ikke endnu. Luk *alle* VS Code- og PowerShell-vinduer og åbn dem igen. Hjælper det ikke, så genstart computeren.

**Du vil hellere bruge pip end uv**
Det kan lade sig gøre: lav et virtuelt miljø og kør `pip install -r requirements.txt`. Vi anbefaler dog uv – så får alle de samme versioner.

**Installationen af uv bliver blokeret**
Prøv i stedet: `winget install --id=astral-sh.uv -e` i PowerShell. Spørg din lærer, hvis det heller ikke virker.

**`ModuleNotFoundError: No module named ...`**
Tjek at terminalen står i projektmappen, og kør `uv sync` igen. Husk at starte scripts med `uv run` – ikke `python`.

**Fejl om `pyproject.toml` eller "No project found"**
Terminalen står ikke i projektmappen. Skriv `cd C:\DDU\machine-learning-yolo` og prøv igen.

**Fejl om at filen ikke findes (`test.webp` eller `aerial-view.mp4`)**
Mappen `examples/input` mangler eller ligger forkert. Tjek mappestrukturen under [Trin 4](#trin-4--hent-projektet-fra-github).

**Webcam: sort billede, eller vinduet lukker med det samme**
- Windows: Gå til **Indstillinger → Beskyttelse af personlige oplysninger og sikkerhed → Kamera** og slå adgang til for apps og skrivebordsapps.
- Luk andre programmer, der bruger kameraet (Teams, Zoom, kamera-appen).
- Har du flere kameraer (fx et eksternt), så ret tallet øverst i `examples/webcam.py`:
  ```python
  CAMERA = 0   # prøv 1, hvis 0 ikke virker
  ```
- **Mac:** Giv VS Code/Terminal adgang til kameraet under *Systemindstillinger → Anonymitet og sikkerhed → Kamera*. Er der en iPhone i nærheden, kan Mac'en vælge iPhonens kamera – prøv et andet tal i `CAMERA`.

**Det kører meget langsomt**
YOLO kører på computerens processor (CPU), så video og webcam kan hakke lidt på ældre computere. Det er helt normalt.

---

## Del 2. Sådan træner man sin egen model

Følg guiden i [train-demo/README.md](train-demo/README.md). Her træner du YOLO til at genkende brikkerne på vores brætspil – og bagefter kan du gøre det samme med dine egne billeder.
