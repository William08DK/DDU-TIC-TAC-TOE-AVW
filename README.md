# Kryds og bolle med webcam

Et fysisk 3 × 3-bræt aflæses med YOLO. Du spiller X og starter. Computeren spiller O og viser sit næste træk med en grøn cirkel på skærmen. Du flytter selv begge spilleres brikker. Computeren vælger træk med minimax.

## Start

Kør fra projektmappen:

```powershell
uv sync
uv run main.py
```

Vælg kamera direkte i spilvinduet med **0–9**. USB-kameraet er ofte **1**, men nummeret afhænger af computeren. Det aktive kameranummer vises nederst. Hvis et kamera ikke virker, beholdes det nuværende. Efter et kameraskift skal brættets fire hjørner markeres igen; spillets træk bevares.

I kan også starte direkte på USB-kameraet med `uv run main.py --camera 1`.
Du kan også installere med `python -m pip install -r requirements.txt` og starte med `python main.py`.

## Sådan spiller I

1. Stil kameraet fast, gerne oven over et tomt bræt med 3 × 3 lige store felter.
2. Klik på brættets fire **ydre** hjørner: øverst venstre, øverst højre, nederst højre, nederst venstre. Programmet retter perspektivet og viser brættet forfra.
3. Læg din X-brik, fjern hænderne, og tryk **mellemrum**, mens spilvinduet er aktivt.
4. Den grønne cirkel viser computerens træk. Læg en O-brik dér, fjern hænderne, og tryk **mellemrum** igen for at bekræfte computerens brik.
5. Fortsæt med din næste X. Vinderen eller uafgjort vises på skærmen.

**R** starter et nyt spil: fjern alle brikker, før du lægger den første X. **K** starter forfra med ny hjørnemarkering, hvis kameraet flyttes. **Q** eller **Esc** lukker.

De gule bogstaver viser sidst godkendte bræt. Ved en fejlaflæsning beholdes det, og du kan trykke mellemrum igen. Programmet kræver ét nyt træk ad gangen; aflæs derfor også efter at have lagt computerens brik.

## Brikker og model

Den medfølgende lærer-model er trænet på figurer, ikke almindeligt tegnede X og O. Som standard bruger vi **maltese** (kors) til dine X og **circle** (cirkel) til computerens O. Brug de fysiske figurer fra datasættet. En model til tegnede kryds og boller skal trænes på billeder af disse.

Andre klassenavne kan vælges sådan:

```powershell
uv run main.py --x-class star --o-class circle
uv run main.py --model train-demo/runs/detect/train-2/weights/best.pt
```

Standardmodellen findes i `train-demo/runs/detect/train/weights/best.pt`. Modelvægte er lokale og følger **ikke** med i Git. På en ny computer skal I kopiere modellen til den sti eller træne den med `uv run train-demo/model-train.py`.

Aflæsningen bruger brikkernes midtpunkter. Hold brikkerne midt i felterne og brug jævnt lys. Ved flere fund i samme felt eller andre figurer afvises aflæsningen. Overses en helt ny brik, så prøv igen; en forkert genkendt brik kan kræve **R** og et nyt spil. Modellen skal afprøves med jeres kamera og 3 × 3-bræt, fordi træningsbillederne kommer fra lærerens 3 × 5-bræt. `--confidence 0.6` ændrer tærsklen for genkendelse (standard: 0.5).

Hvis kameraet ikke åbner, prøv `--camera 1`, luk andre kamera-apps og kontrollér Windows' kameratilladelser.

## Filer

- `main.py`: webcam, hjørnemarkering, YOLO og visning.
- `game.py`: regler, turstyring og computerens træk.
- `train-demo/`: træningsscript, datasæt og eksisterende lokale modeller.
- `CUT-filer/`: den eksisterende SVG-tegning, bevaret som materiale til brættet.
- `tests/`: kontrol af spillogik og aflæsning. Kør `uv run python -m unittest discover -s tests`.

De generelle billed-/videodemoer, mobilserveren og COCO-dokumentationen er fjernet.
