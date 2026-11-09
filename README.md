# Kryds og bolle med webcam

Et fysisk 3 × 3-bræt aflæses med YOLO. Du spiller X og starter. Computeren spiller O og viser sit næste træk med en grøn cirkel på skærmen. Du flytter selv begge spilleres brikker. Computeren vælger træk med minimax.

Computeren kan ikke tabe fra et tomt bræt ved lovligt spil: den gennemregner alle mulige fortsættelser, vælger en sikker sejr, hvis muligt, og sikrer ellers uafgjort. Den vælger den hurtigste sikre sejr. Det kræver intet datasæt til strategi; YOLO-modellen skal kun genkende brikkerne korrekt. Garantien gælder det aflæste bræt og forudsætter, at computerens brikker placeres i de foreslåede felter.

## Start

Kør fra projektmappen:

```powershell
uv sync
uv run main.py
```

Programmet starter på jeres USB-kamera **1**. Vælg et andet kamera i menuen til højre, og klik **Skift**. Genvejene **0–9** virker også. Numrene afhænger af computeren. Det aktive kameranummer vises øverst. Hvis et kamera ikke virker, beholdes det nuværende. Efter et kameraskift skal brættets fire hjørner markeres igen; spillets træk bevares.

I kan også starte direkte på USB-kameraet med `uv run main.py --camera 1`.
Du kan også installere med `python -m pip install -r requirements.txt` og starte med `python main.py`.

## Rotation, spejlvending og debug

**Rotér 90°** drejer kamerabilledet med uret. **Spejlvend** vender det vandret; klik igen for at slå spejling fra. Vælg orienteringen inden spillet: ændringer starter et nyt spil og kræver, at brættet tømmes og hjørnerne markeres igen. Genkendelsen bruger samme orientering som visningen.

**Debug · live** åbner et separat vindue med løbende genkendelse: kasser omkring alle fund, klassenavne og modellens sikkerhed i procent. Spilbrikker vises også som **Kryds / X** eller **Bolle / O**, hvis de matcher de valgte brikklasser. Debug godkender ikke træk; brug fortsat Aflæs brættet i spillet.

- **Spilbrikker** bruger spillets valgte model og viser alle dens klasser, også andre figurer end X/O.
- **Hverdagsting** bruger den lokale generelle model `yolo26n.pt`, der kan genkende sine trænede kategorier af hverdagsting. Den kan ikke genkende vilkårlige objekter eller nye speciallavede brikker uden relevant træning. En anden generel model kan vælges med `--debug-model "STI/TIL/model.pt"`.

Før hjørnemarkering analyserer debug hele kamerabilledet; derefter analyserer det det udrettede bræt. Kameravideoen opdateres uafhængigt af genkendelsen, der kører i baggrunden uden kø af gamle billeder. Kasserne viser seneste analyse og skjules, når analysebilledet er over et sekund gammelt; de kan derfor halte lidt efter objekter i bevægelse. Analysetiden vises i debugvinduet.

Debug bruger som standard 320 pixels til analysen for at spare regnekraft. Små detaljer kan genkendes bedre med `--debug-imgsz 640`, som også kræver mere regnetid; 480 er et mellemtrin. Det ændrer ikke opløsningen ved spillets almindelige aflæsning. Hastigheden afhænger af computeren. Hverdagsting kræver, at den generelle model findes lokalt; programmet henter den ikke automatisk.

## Sådan spiller I

1. Stil kameraet fast, gerne oven over et tomt bræt med 3 × 3 lige store felter.
2. Klik på brættets fire **ydre** hjørner: øverst venstre, øverst højre, nederst højre, nederst venstre. Programmet retter perspektivet og viser brættet forfra.
3. Læg din X-brik, fjern hænderne, og klik **Aflæs brættet** eller tryk **mellemrum**.
4. Den grønne cirkel viser computerens træk. Læg en O-brik dér, fjern hænderne, og klik **Bekræft O-brikken** eller tryk **mellemrum** igen.
5. Fortsæt med din næste X. Vinderen eller uafgjort vises på skærmen.

**Nyt spil** / **R** starter et nyt spil: fjern alle brikker, før du lægger den første X. **Markér hjørner igen** / **K** lader jer markere brættet igen, hvis kameraet flyttes. Spillets træk bevares ved ny hjørnemarkering. **Q** eller **Esc** lukker.

Det lille bræt til højre viser sidst godkendte træk. En stiplet grøn cirkel viser, hvor computerens næste brik skal placeres. Ved en fejlaflæsning beholdes tilstanden, og du kan aflæse igen. Programmet kræver ét nyt træk ad gangen; aflæs derfor også efter at have lagt computerens brik.

## Brikker og model

Ved spillets aflæsning genkendes brikkerne i det fulde, orienterede kamerabillede. Deres midtpunkter overføres derefter til det markerede 3 × 3-bræt. Fund uden for brættet ignoreres. Det undgår, at perspektivopretningen og forstørrelsen af brikkerne forringer modellens genkendelse.

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

- `main.py`: webcam, hjørnemarkering og YOLO.
- `ui.py`: spilvindue, kameravælger, knapper og overblik.
- `vision.py`: rotation, spejlvending og debugmarkeringer.
- `game.py`: regler, turstyring og computerens træk.
- `train-demo/`: træningsscript, datasæt og eksisterende lokale modeller.
- `CUT-filer/`: den eksisterende SVG-tegning, bevaret som materiale til brættet.
- `tests/`: kontrol af spillogik og aflæsning. Kør `uv run python -m unittest discover -s tests`.

De generelle billed-/videodemoer, mobilserveren og COCO-dokumentationen er fjernet.
